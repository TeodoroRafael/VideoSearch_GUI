import threading
from typing import Any, Dict, Optional

from transformers import (
    AutoProcessor,
    CLIPModel,
    LlavaForConditionalGeneration,
    SiglipModel,
)

from models.clip import CLIPWrapper
from models.llava import LLaVaWrapper
from models.siglip import SigLipWrapper

# model_id defines the default model_id that can be overwritten
CONFIGS = {
    "clip": {
        "model_id": "openai/clip-vit-base-patch32",
        "model_class": CLIPModel,
        "processor_class": AutoProcessor,
        "wrapper_class": CLIPWrapper,
    },
    "siglip": {
        "model_id": "google/siglip-base-patch16-256",
        "model_class": SiglipModel,
        "processor_class": AutoProcessor,
        "wrapper_class": SigLipWrapper,
    },
    "llava": {
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "model_class": LlavaForConditionalGeneration,
        "processor_class": AutoProcessor,
        "wrapper_class": LLaVaWrapper,
    },
}


def get_model_config(
        model_family: str,
        model_id: Optional[str] = None,
) -> Dict[str, Any]:
    config = CONFIGS.get(model_family, {})
    if not config:
        raise ValueError(f"Model config is not parsed. Plase use model_family from {list(CONFIGS.keys())}")
    if model_id is not None:
        config["model_id"] = model_id
    return config


# Loaded (model, processor) pairs are expensive (a multi-second from_pretrained
# call each) and safe to reuse read-only across requests, so keep one instance
# per (model_family, model_id, device) alive for the life of the process
# instead of reloading it on every call site. The lock serializes concurrent
# first-time loads of the *same* entry so two requests don't race to build it
# twice; already-cached lookups just return the cached wrapper.
_VLM_WRAPPER_CACHE: Dict[Any, Any] = {}
_VLM_WRAPPER_CACHE_LOCK = threading.Lock()


def load_vlm_wrapper(
        model_family: str,
        model_id: Optional[str] = None,
        device: str = "cpu",
) -> Any:
    with _VLM_WRAPPER_CACHE_LOCK:
        config = get_model_config(model_family, model_id)
        cache_key = (model_family, config["model_id"], device)

        wrapper = _VLM_WRAPPER_CACHE.get(cache_key)
        if wrapper is None:
            processor = config["processor_class"].from_pretrained(config["model_id"])
            model = config["model_class"].from_pretrained(config["model_id"])
            model.to(device)
            model.eval()
            wrapper = config["wrapper_class"](model=model, processor=processor)
            _VLM_WRAPPER_CACHE[cache_key] = wrapper

        return wrapper
