import argparse
import glob
import json
import os
from typing import List

import cv2
from tqdm import tqdm


def find_shot_boundaries_json(video_path: str) -> str:
    # The json filename doesn't always match the video's stem exactly
    # (spaces vs. underscores), so look for the suffix in the video's folder.
    video_dir = os.path.dirname(os.path.abspath(video_path))
    matches = glob.glob(os.path.join(video_dir, "*_shot_boundaries_datamodel.json"))
    if not matches:
        raise FileNotFoundError(f"No *_shot_boundaries_datamodel.json found in {video_dir}")
    return matches[0]


def load_dimension_indices(json_path: str) -> List[int]:
    with open(json_path, "r") as f:
        shot_data = json.load(f)
    return [entry["dimension_idx"] for entry in shot_data["data"]]


def frames_dir_for_video(video_path: str) -> str:
    video_dir = os.path.dirname(os.path.abspath(video_path))
    return os.path.join(video_dir, "frames")


def has_extracted_frames(video_path: str) -> bool:
    frames_dir = frames_dir_for_video(video_path)
    return os.path.isdir(frames_dir) and bool(glob.glob(os.path.join(frames_dir, "*.jpg")))


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    try:
        return cap.get(cv2.CAP_PROP_FPS) or 0.0
    finally:
        cap.release()


def extract_frames(video_path: str, frame_indices: List[int], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    try:
        for frame_idx in tqdm(frame_indices, desc="Extracting frames"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if not success:
                print(f"Warning: could not read frame {frame_idx}")
                continue
            out_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
    finally:
        cap.release()


def extract_frames_from_video(video_path: str) -> str:
    """
    Reads the `<video_name>_shot_boundaries_datamodel.json` next to `video_path`
    and saves one .jpg per `dimension_idx` into a `frames/` folder alongside
    the video. Returns the output folder path.
    """
    json_path = find_shot_boundaries_json(video_path)
    frame_indices = load_dimension_indices(json_path)

    output_dir = frames_dir_for_video(video_path)
    extract_frames(video_path, frame_indices, output_dir)
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract shot-boundary frames from a video as .jpg images"
    )
    parser.add_argument("video_path", type=str, help="Path to the video file")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = extract_frames_from_video(args.video_path)
    print(f"Frames saved to {output_dir}")


if __name__ == "__main__":
    main()
