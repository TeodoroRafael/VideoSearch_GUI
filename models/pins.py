import json
import os
from typing import Any, Dict, List

from models.frame_extractor import get_video_fps


def pins_path(video_path: str, video_name: str) -> str:
    video_dir = os.path.dirname(os.path.abspath(video_path))
    return os.path.join(video_dir, f"{video_name}_pins.json")


def load_pins(video_path: str, video_name: str) -> List[Dict[str, Any]]:
    """
    Reads `<video_name>_pins.json` next to the video — the markers "memory"
    file. Creates it, empty, the first time a video is loaded.
    """
    path = pins_path(video_path, video_name)
    if not os.path.isfile(path):
        save_pins(video_path, video_name, [])
        return []
    with open(path, "r") as f:
        return json.load(f)


def save_pins(video_path: str, video_name: str, pins: List[Dict[str, Any]]) -> None:
    with open(pins_path(video_path, video_name), "w") as f:
        json.dump(pins, f, indent=2)


def seconds_to_timestamp(seconds: float) -> str:
    total = max(int(round(seconds)), 0)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_pin_record(video_path: str, pin_id: int, time_seconds: float, label: str) -> Dict[str, Any]:
    # frame_number is the pin's time converted to a frame index (time * fps);
    # the stored "time" is just that same moment formatted as HH:MM:SS.
    fps = get_video_fps(video_path)
    frame_number = int(round(time_seconds * fps)) if fps else 0
    return {
        "id": pin_id,
        "frame_number": frame_number,
        "time": seconds_to_timestamp(time_seconds),
        "label": label,
    }


def pin_time_seconds(pin: Dict[str, Any], fps: float) -> float:
    # Reconstructed from frame_number (not re-parsed from the "HH:MM:SS"
    # string) so seeking back to a restored pin stays frame-accurate.
    return (pin["frame_number"] / fps) if fps else 0.0


def upsert_pin(video_path: str, video_name: str, pin: Dict[str, Any]) -> List[Dict[str, Any]]:
    pins = [p for p in load_pins(video_path, video_name) if p["id"] != pin["id"]]
    pins.append(pin)
    pins.sort(key=lambda p: p["frame_number"])
    save_pins(video_path, video_name, pins)
    return pins


def delete_pin(video_path: str, video_name: str, pin_id: int) -> List[Dict[str, Any]]:
    pins = [p for p in load_pins(video_path, video_name) if p["id"] != pin_id]
    save_pins(video_path, video_name, pins)
    return pins
