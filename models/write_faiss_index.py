import argparse
import hashlib
import glob
import json
import os
from pathlib import Path

import numpy as np
# torch must be imported before faiss: on macOS, loading faiss's bundled
# OpenMP runtime first corrupts torch's own thread pool init and segfaults
# on the first model forward pass.
import torch
import faiss
from PIL import Image
from tqdm import tqdm

from models.configs import get_model_config, load_vlm_wrapper


def parse_args():
    # CLI arguments define where images are loaded from, which model encodes them,
    # and where the resulting FAISS artifacts are saved.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to dataset with images')
    parser.add_argument('--output', type=str, required=True, help='Path to output FAISS index')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model_family', type=str, required=True, help='VLM model family')
    parser.add_argument('--model_id', type=str, required=True, help='HF model id')
    parser.add_argument('--index_type', type=str, default='flat_ip', help='Index type')
    parser.add_argument(
        '--id_source',
        type=str,
        default='relative_path',
        choices=['sequential', 'filename', 'stem', 'relative_path'],
        help='How to derive image labels used for FAISS IDs'
    )
    parser.add_argument(
        '--m',
        type=int,
        default=32,
        help='Number of connections per layer only for `hnsw` index type.'
    )
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device cuda/cpu for generating embeddings')
    return parser.parse_args()


def get_image_paths(data_dir):
    # Recursively gather common image formats from the dataset root.
    extensions = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(data_dir, f'**/*.{ext}'), recursive=True))
    return image_paths


def encode_images(vlm_wrapper, image_paths, batch_size):
    # This function converts all images into dense embedding vectors.
    features = []
    paths = []
    
    # Process images in mini-batches to control memory usage.
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        valid_paths = []
        
        for path in tqdm(batch_paths, desc="Encoding images"):
            try:
                # Convert to RGB so the model receives a consistent image format.
                img = Image.open(path).convert('RGB')
                batch_images.append(img)
                valid_paths.append(path)
            except Exception as e:
                # Keep indexing robust: skip unreadable/corrupt files and continue.
                print(f"Error processing {path}: {e}")
        # Wrapper-specific preprocessing (resize/normalize/tensor conversion).
        processed_images = vlm_wrapper.process_inputs(images=batch_images)
        with torch.no_grad():
            # Disable gradients because this is inference-only embedding extraction.
            outputs = vlm_wrapper.get_image_embeddings(processed_images)
            features.append(outputs.cpu().numpy())
        # Persist only the paths that produced embeddings so ids stay aligned.
        paths.extend(valid_paths)
    
    return np.vstack(features), paths


def build_image_labels(image_paths, data_root, id_source='relative_path'):
    # Convert each file path into the human-readable label that will be attached to the FAISS id.
    data_root = Path(data_root).resolve()
    labels = []
    for path in image_paths:
        path_obj = Path(path).resolve()
        if id_source == 'filename':
            labels.append(path_obj.name)
        elif id_source == 'stem':
            labels.append(path_obj.stem)
        elif id_source == 'relative_path':
            labels.append(str(path_obj.relative_to(data_root)))
        else:
            raise ValueError(f"Invalid id_source: {id_source}")
    return labels


def labels_to_faiss_ids(labels):
    # FAISS ids must be int64, so we derive stable 64-bit ids from the chosen label string.
    ids = []
    seen_ids = set()
    for label in labels:
        digest = hashlib.blake2b(label.encode('utf-8'), digest_size=8).digest()
        faiss_id = int.from_bytes(digest, byteorder='little', signed=False) & np.iinfo(np.int64).max
        if faiss_id in seen_ids:
            raise ValueError(
                f"Duplicate FAISS id generated for label '{label}'. Use a more unique id_source, such as 'relative_path'."
            )
        seen_ids.add(faiss_id)
        ids.append(faiss_id)
    return np.asarray(ids, dtype=np.int64)


def create_faiss_index(features, feature_dim, index_type='flat_ip', m=32, ids=None):
    # L2-normalize vectors so inner product behaves like cosine similarity.
    faiss.normalize_L2(features)
    
    if index_type == 'flat_ip':
        # Exact search: highest quality retrieval, more compute at query time.
        base_index = faiss.IndexFlatIP(feature_dim)
    elif index_type == 'hnsw':
        # Approximate search graph: faster queries on larger collections.
        base_index = faiss.IndexHNSWFlat(feature_dim, m)
    else:
        raise ValueError(f"Invalid index type: {index_type}")
    
    if ids is not None:
        index = faiss.IndexIDMap2(base_index)
        index.add_with_ids(features, ids)
    else:
        index = base_index
        index.add(features)
    
    return index


def build_index(
    data: str,
    output: str,
    model_family: str,
    model_id: str,
    batch_size: int = 32,
    index_type: str = 'flat_ip',
    id_source: str = 'relative_path',
    m: int = 32,
    device: str = None,
):
    """
    Encode every image under `data` with the given model and write a FAISS
    index (plus label/path lookup files) under `output/model_id/`.

    This holds the logic `main()` used to run inline, so it can also be
    called directly from Python (see `build_coco_index` below) instead of
    only via the CLI.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get all image paths
    image_paths = get_image_paths(data)
    print(f"Found {len(image_paths)} images")

    # Resolve model classes (processor/model/wrapper) from configured family + id.
    model_config = get_model_config(model_family, model_id)

    # Build Hugging Face processor/model, then adapt them through a wrapper API.
    processor = model_config["processor_class"].from_pretrained(model_config["model_id"])
    model = model_config["model_class"].from_pretrained(model_config["model_id"])
    wrapper = model_config["wrapper_class"](model=model, processor=processor)

    # Move model to the selected device and switch to inference mode.
    model.to(device)
    model.eval()

    features, paths = encode_images(wrapper, image_paths, batch_size)
    feature_dim = features.shape[1]

    labels = build_image_labels(paths, data, id_source)
    faiss_ids = labels_to_faiss_ids(labels) if id_source != 'sequential' else np.arange(len(paths), dtype=np.int64)

    # Create FAISS index
    index = create_faiss_index(features, feature_dim, index_type, m, faiss_ids)

    # Save index and paths
    output_dir = os.path.join(output, model_id)
    os.makedirs(output_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(output_dir, "image_index.faiss"))

    # Save a lookup table so FAISS ids can be translated back to readable labels and paths.
    id_map_path = os.path.join(output_dir, "image_id_map.json")
    with open(id_map_path, "w") as f:
        json.dump(
            [
                {
                    "faiss_id": int(faiss_id),
                    "label": label,
                    "path": path,
                }
                for faiss_id, label, path in zip(faiss_ids.tolist(), labels, paths)
            ],
            f,
            indent=2,
        )

    # Keep the legacy positional path file for older readers.
    with open(os.path.join(output_dir, "image_paths.txt"), "w") as f:
        for path in paths:
            f.write(f"{path}\n")

    print(f"Index created with {len(paths)} images and saved to {output_dir}")
    return output_dir


def faiss_index_path(video_path: str, video_name: str) -> str:
    video_dir = os.path.dirname(os.path.abspath(video_path))
    return os.path.join(video_dir, f"{video_name}.faiss")


def has_faiss_index(video_path: str, video_name: str) -> bool:
    return os.path.isfile(faiss_index_path(video_path, video_name))


def build_index_for_frames(
    video_path: str,
    frames_dir: str,
    video_name: str,
    model_family: str = 'clip',
    model_id: str = None,
    batch_size: int = 32,
    device: str = None,
) -> str:
    """
    Encode every image in `frames_dir` (the shot-boundary frames extracted
    for `video_path`) and save the resulting FAISS index as
    `<video_name>.faiss` next to the video (and its shot-boundaries json),
    not inside `frames_dir`. Returns the index path.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image_paths = get_image_paths(frames_dir)

    # Cached across calls (shared with models/search.py) — see
    # load_vlm_wrapper() — so a search right after Create FAISS reuses the
    # model this just loaded instead of loading it again.
    wrapper = load_vlm_wrapper(model_family, model_id, device)

    features, paths = encode_images(wrapper, image_paths, batch_size)
    feature_dim = features.shape[1]

    labels = build_image_labels(paths, frames_dir, id_source='filename')
    faiss_ids = labels_to_faiss_ids(labels)

    index = create_faiss_index(features, feature_dim, index_type='flat_ip', ids=faiss_ids)

    index_path = faiss_index_path(video_path, video_name)
    faiss.write_index(index, index_path)

    video_dir = os.path.dirname(os.path.abspath(video_path))
    id_map_path = os.path.join(video_dir, f"{video_name}_id_map.json")
    with open(id_map_path, "w") as f:
        json.dump(
            [
                {"faiss_id": int(faiss_id), "label": label, "path": path}
                for faiss_id, label, path in zip(faiss_ids.tolist(), labels, paths)
            ],
            f,
            indent=2,
        )

    print(f"Index created with {len(paths)} images and saved to {index_path}")
    return index_path


def build_coco_index():
    """
    Equivalent to running:

        python write_faiss_index.py \\
            --data data/coco/test2014 \\
            --output faiss/coco/ \\
            --batch_size 64 \\
            --model_family clip \\
            --model_id openai/clip-vit-large-patch14

    Handy for calling from a notebook or another script without retyping
    the CLI invocation.
    """
    return build_index(
        data='data/coco/test2014',
        output='faiss/coco/',
        batch_size=64,
        model_family='clip',
        model_id='openai/clip-vit-large-patch14',
    )


def main():
    args = parse_args()
    build_index(
        data=args.data,
        output=args.output,
        model_family=args.model_family,
        model_id=args.model_id,
        batch_size=args.batch_size,
        index_type=args.index_type,
        id_source=args.id_source,
        m=args.m,
        device=args.device,
    )

if __name__ == "__main__":
    main()