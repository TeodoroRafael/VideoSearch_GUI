"""
Phase 2 checkpoint: prove that relevance feedback (CaptionVLMRelevanceFeedback +
RocchioUpdate) moves a text query's embedding closer to images the user marks
Relevant and further from images marked Irrelevant.

Mirrors the app's "Select frames" -> "Feedback" flow: run an initial text
search (embedding + cosine similarity, same idea as vlm_checkpoint.py) over
ten bundled scikit-image sample photos standing in for ten retrieved
results, mark two of them Relevant and five Irrelevant (the rest left
unannotated, as if never selected), caption the annotated ones with the
captioning VLM, and fold those captions back into the query embedding via
Rocchio's algorithm.

Uses bundled scikit-image sample photos, so it runs fully offline aside from
the retrieval/captioning model weights.
"""

import os
import sys
import tempfile
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from skimage import data
from transformers import AutoProcessor, LlavaForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.clip import CLIPWrapper
from models.llava import LLaVaWrapper
from models.relevance_feedback import CaptionVLMRelevanceFeedback, RocchioUpdate

# A small LLaVA variant (same LlavaForConditionalGeneration architecture the
# wrapper expects) so the checkpoint doesn't have to pull down the default
# 7B captioning weights just to prove the pipeline works.
CAPTIONING_MODEL_ID = "llava-hf/llava-interleave-qwen-0.5b-hf"


def load_captioning_wrapper(device: torch.device) -> LLaVaWrapper:
    model = LlavaForConditionalGeneration.from_pretrained(CAPTIONING_MODEL_ID)
    processor = AutoProcessor.from_pretrained(CAPTIONING_MODEL_ID)
    return LLaVaWrapper(model=model, processor=processor)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def whole_image_box(img_size: int, label: str) -> List[Dict[str, Any]]:
    """A single annotation box covering the whole image, matching the shape
    the GUI's box-annotator produces for a fragment selection."""
    return [{"xmin": 0, "ymin": 0, "xmax": img_size, "ymax": img_size, "label": label}]


def label_tag(label: str) -> str:
    if label in POSITIVE_LABELS:
        return "Relevant"
    if label in NEGATIVE_LABELS:
        return "Irrelevant"
    return "-"


def rank_results(similarity: torch.Tensor, ordered_labels: List[str]) -> List[Dict[str, Any]]:
    """Sort labels by similarity score, descending -- i.e. the ranked search
    results a user would actually see, best match first."""
    order = torch.argsort(similarity, descending=True).tolist()
    return [
        {"rank": rank + 1, "label": ordered_labels[idx], "score": similarity[idx].item()}
        for rank, idx in enumerate(order)
    ]


def print_ranking(title: str, similarity: torch.Tensor, ordered_labels: List[str]) -> None:
    print(f"\n{title}")
    for entry in rank_results(similarity, ordered_labels):
        print(
            f"  #{entry['rank']:<2d} {entry['label']:18s} {entry['score']:.4f}  "
            f"{label_tag(entry['label'])}"
        )


def print_rank_comparison(
    ordered_labels: List[str],
    initial_similarity: torch.Tensor,
    updated_similarity: torch.Tensor,
) -> None:
    """Show, per image, how its rank in the search results moved once the
    query was updated with relevance feedback -- the "before vs after" view
    of the search itself, not just raw similarity scores."""
    initial_rank = {e["label"]: e["rank"] for e in rank_results(initial_similarity, ordered_labels)}
    updated_rank = {e["label"]: e["rank"] for e in rank_results(updated_similarity, ordered_labels)}

    print("\nSearch rank before -> after feedback:")
    for label in sorted(ordered_labels, key=lambda l: updated_rank[l]):
        before, after = initial_rank[label], updated_rank[label]
        delta = before - after  # positive: moved up (toward #1)
        movement = f"up {delta}" if delta > 0 else f"down {-delta}" if delta < 0 else "unchanged"
        print(f"  {label:18s} {label_tag(label):10s} #{before:<2d} -> #{after:<2d}  ({movement})")


def print_final_message(
    query: str,
    ordered_labels: List[str],
    initial_similarity: torch.Tensor,
    updated_similarity: torch.Tensor,
    relevant_captions: List[str],
) -> None:
    """Summarize, in plain language, what the updated query now represents
    best -- not just the numbers, but which search result actually wins and
    why, so it's clear how the feedback changed the search."""
    before = rank_results(initial_similarity, ordered_labels)[0]
    after = rank_results(updated_similarity, ordered_labels)[0]
    caption_summary = "; ".join(f'"{c}"' for c in relevant_captions) if relevant_captions else "no captions"

    print("\n" + "=" * 60)
    print("Final query update")
    print("=" * 60)
    print(f"Original query:      {query!r}")
    print(f"Best match before:   {before['label']} (score {before['score']:.4f}, {label_tag(before['label'])})")
    print(f"Best match after:    {after['label']} (score {after['score']:.4f}, {label_tag(after['label'])})")
    print(f"Learned from:        {caption_summary}")

    if after["label"] in POSITIVE_LABELS and before["label"] not in POSITIVE_LABELS:
        print(
            f"\nThe updated query now semantically means {query!r} the way the user "
            f"clarified it through feedback -- closer to {caption_summary} -- so the top "
            f"search result flipped from the off-target '{before['label']}' to the "
            f"user-confirmed '{after['label']}'."
        )
    elif after["label"] in POSITIVE_LABELS:
        print(
            f"\nThe updated query stayed anchored on a Relevant result ('{after['label']}'), "
            "now pulled further ahead of the Irrelevant examples the user ruled out."
        )
    else:
        print(
            f"\nThe updated query's top match ('{after['label']}') still isn't one of the "
            "images marked Relevant -- feedback narrowed the gap but didn't flip the top rank."
        )


def load_sample_image(name: str) -> Image.Image:
    """Load a bundled scikit-image sample by name as an RGB PIL image (some
    samples are grayscale or boolean masks)."""
    array = getattr(data, name)()
    if array.dtype == bool:
        array = array.astype("uint8") * 255
    return Image.fromarray(array).convert("RGB")


# Ten bundled scikit-image sample photos standing in for ten retrieved search
# results: two the user marks Relevant, five they mark Irrelevant, and three
# left unannotated (as if never selected at all).
POSITIVE_LABELS = ["astronaut", "hubble_deep_field"]
NEGATIVE_LABELS = ["rocket", "coffee", "camera", "coins", "brick"]
UNANNOTATED_LABELS = ["chelsea", "moon", "horse"]


def main():
    device = get_device()
    print(f"Using device: {device}")

    retrieval_wrapper = CLIPWrapper()
    retrieval_wrapper.model.to(device)
    retrieval_wrapper.model.eval()

    captioning_wrapper = load_captioning_wrapper(device)
    captioning_wrapper.model.to(device)
    captioning_wrapper.model.eval()

    feedback = CaptionVLMRelevanceFeedback(
        vlm_wrapper_retrieval=retrieval_wrapper,
        vlm_wrapper_captioning=captioning_wrapper,
    )
    rocchio_update = RocchioUpdate(alpha=0.8, beta=0.3, gamma=0.3)

    query = "space"

    ordered_labels = POSITIVE_LABELS + NEGATIVE_LABELS + UNANNOTATED_LABELS
    sample_images = {label: load_sample_image(label) for label in ordered_labels}

    # Step 1: initial text search over the retrieved images, same idea as
    # vlm_checkpoint.py -- cosine similarity between the query and each image
    # embedding.
    image_inputs = retrieval_wrapper.process_inputs(
        images=[sample_images[label] for label in ordered_labels]
    )
    text_inputs = retrieval_wrapper.process_inputs(text=[query])
    with torch.no_grad():
        image_embeds = retrieval_wrapper.get_image_embeddings(image_inputs)
        query_embeds = retrieval_wrapper.get_text_embeddings(text_inputs)
    image_embeds = F.normalize(image_embeds, p=2, dim=-1)
    query_embeds = F.normalize(query_embeds, p=2, dim=-1)

    initial_similarity = (query_embeds @ image_embeds.T).squeeze(0)
    print(f"\nQuery: {query!r}")
    print_ranking("Search results BEFORE feedback:", initial_similarity, ordered_labels)

    # Step 2: simulate the user selecting retrieved frames and giving
    # feedback -- astronaut/hubble_deep_field are what they were after,
    # rocket/coffee/camera/coins/brick are near-misses or off-topic results
    # they rule out, and the rest are left unannotated as if never selected.
    with tempfile.TemporaryDirectory() as tmp_dir:
        relevant_image_paths = []
        for label in ordered_labels:
            path = os.path.join(tmp_dir, f"{label}.jpg")
            sample_images[label].save(path)
            relevant_image_paths.append(path)

        annotator_json_boxes_list = [
            whole_image_box(feedback.img_size, "Relevant") if label in POSITIVE_LABELS
            else whole_image_box(feedback.img_size, "Irrelevant") if label in NEGATIVE_LABELS
            else None
            for label in ordered_labels
        ]

        result = feedback(
            query=query,
            relevant_image_paths=relevant_image_paths,
            annotator_json_boxes_list=annotator_json_boxes_list,
            prompt_based_on_query=True,
            top_k_feedback=len(relevant_image_paths),
        )

    print(f"\nRelevant captions:   {result['relevant_captions']}")
    print(f"Irrelevant captions: {result['irrelevant_captions']}")

    # Step 3: fold the caption embeddings back into the query embedding.
    updated_query_embeds = rocchio_update(
        query_embeddings=query_embeds,
        positive_embeddings=result["positive"],
        negative_embeddings=result["negative"],
    )

    updated_similarity = (updated_query_embeds @ image_embeds.T).squeeze(0)
    print_ranking("Search results AFTER feedback:", updated_similarity, ordered_labels)
    print_rank_comparison(ordered_labels, initial_similarity, updated_similarity)
    print_final_message(
        query, ordered_labels, initial_similarity, updated_similarity, result["relevant_captions"]
    )

    positive_idx = [ordered_labels.index(label) for label in POSITIVE_LABELS]
    negative_idx = [ordered_labels.index(label) for label in NEGATIVE_LABELS]

    # The absolute similarity to any single image can shift either way once
    # real (imperfect) generated captions are folded in -- what relevance
    # feedback actually promises is a *relative* correction between the two
    # groups: on average, the images marked Relevant should end up closer to
    # the query than the images marked Irrelevant, more so than they started.
    initial_pos = initial_similarity[positive_idx].mean().item()
    initial_neg = initial_similarity[negative_idx].mean().item()
    updated_pos = updated_similarity[positive_idx].mean().item()
    updated_neg = updated_similarity[negative_idx].mean().item()
    initial_gap = initial_pos - initial_neg
    updated_gap = updated_pos - updated_neg

    print(f"\nMean Relevant similarity:   {initial_pos:.4f} -> {updated_pos:.4f}")
    print(f"Mean Irrelevant similarity: {initial_neg:.4f} -> {updated_neg:.4f}")
    print(f"Relevant-vs-Irrelevant gap: {initial_gap:.4f} -> {updated_gap:.4f}")

    assert updated_gap > initial_gap, (
        "Feedback should widen the average similarity gap between the "
        "Relevant and Irrelevant image groups, relative to the initial query."
    )
    assert updated_pos > updated_neg, (
        "After feedback, the Relevant image group should, on average, rank "
        "closer to the query than the Irrelevant image group."
    )
    print(
        "\nCheckpoint passed: relevance feedback widened the gap between the "
        "Relevant and Irrelevant image groups enough to fix the ranking."
    )


if __name__ == "__main__":
    main()
