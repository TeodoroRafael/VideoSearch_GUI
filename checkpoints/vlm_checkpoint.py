"""
Phase 1 checkpoint: prove that CLIPWrapper embeds text and images into the
same vector space, so cosine similarity between a caption and its matching
image is higher than between the caption and an unrelated image.

Uses two bundled scikit-image sample photos, so it runs fully offline.
"""

import torch
import torch.nn.functional as F
from PIL import Image
from skimage import data

from models.clip import CLIPWrapper


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = get_device()
    print(f"Using device: {device}")

    wrapper = CLIPWrapper()
    wrapper.model.to(device)
    wrapper.model.eval()

    images = [
        Image.fromarray(data.astronaut()),
        Image.fromarray(data.chelsea()),
    ]
    image_labels = ["astronaut", "cat"]

    texts = [
        "a photo of an astronaut in a spacesuit",
        "a photo of a cat",
    ]

    image_inputs = wrapper.process_inputs(images=images)
    text_inputs = wrapper.process_inputs(text=texts)

    with torch.no_grad():
        image_embeds = wrapper.get_image_embeddings(image_inputs)
        text_embeds = wrapper.get_text_embeddings(text_inputs)

    image_embeds = F.normalize(image_embeds, p=2, dim=-1)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)

    similarity = text_embeds @ image_embeds.T

    print(f"\nimage_embeds shape: {tuple(image_embeds.shape)}")
    print(f"text_embeds shape:  {tuple(text_embeds.shape)}\n")

    print(f"{'':30s}" + "".join(f"{label:>12s}" for label in image_labels))
    for i, text in enumerate(texts):
        row = "".join(f"{similarity[i, j].item():12.4f}" for j in range(len(images)))
        print(f"{text:30s}{row}")

    best_match = similarity.argmax(dim=1).tolist()
    assert best_match == list(range(len(texts))), (
        f"Expected each caption to best match its own image, got matches: {best_match}"
    )
    print("\nCheckpoint passed: each caption is most similar to its matching image.")


if __name__ == "__main__":
    main()
