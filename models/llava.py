from dataclasses import dataclass, field
from typing import Any, Dict

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from models.utils import bitsandbytes_8bit_config
from models.vlm_wrapper import VLMWrapperCaptioning


def init_llava(
        model_config: Dict[str, Any],
        device: str = "cuda",
        use_8bit: bool = False
    ):
    device_type = device.type if isinstance(device, torch.device) else str(device)

    quantization_config = None
    if use_8bit and device_type == "cuda":
        quantization_config = bitsandbytes_8bit_config()
    elif use_8bit:
        print(
            f"Warning: 8-bit quantization requires CUDA, but device is '{device_type}'. "
            "Falling back to float16."
        )

    # bitsandbytes quantization already stores weights at low precision; only
    # request fp16 when loading full-precision weights on an accelerator.
    torch_dtype = torch.float16 if (quantization_config is None and device_type != "cpu") else None

    try:
        model = model_config["model_class"].from_pretrained(
            model_config["model_id"],
            quantization_config=quantization_config,
            torch_dtype=torch_dtype
        )
    except (RuntimeError, AttributeError) as exc:
        if use_8bit:
            print(
                "Warning: 8-bit LLaVA loading failed, falling back to standard precision. "
                f"Reason: {exc}"
            )
            model = model_config["model_class"].from_pretrained(
                model_config["model_id"],
                torch_dtype=torch_dtype
            )
        else:
            raise

    if quantization_config is None:
        model = model.to(device)
    processor = model_config["processor_class"].from_pretrained(model_config["model_id"])

    vlm_wrapper = model_config["wrapper_class"](model=model, processor=processor)
    return vlm_wrapper

@dataclass
class LLaVaWrapper(VLMWrapperCaptioning):
    model: Any = field(
        default_factory=lambda: LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            device_map={"": 0},
            torch_dtype=torch.float16
        )
    )
    processor: Any = field(
        default_factory=lambda: AutoProcessor.from_pretrained(
            "llava-hf/llava-1.5-7b-hf"
        )
    )

    def __post_init__(self):
        self.processor.tokenizer.padding_side = "left"

    def process_inputs(self, apply_template=True, **kwargs):
        required_keys = {'image', 'prompt'}
        if not required_keys.issubset(kwargs.keys()):
            raise ValueError(f"Missing required arguments: {required_keys - set(kwargs.keys())}")

        if apply_template:
            # Uses the processor's own chat template, so the prompt format
            # matches whatever backbone (Vicuna, Qwen, ...) the model was
            # trained with instead of assuming a fixed "USER: ... ASSISTANT:" format.
            prompts = [
                self.processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
                    add_generation_prompt=True
                )
                for prompt in kwargs['prompt']
            ]
        else:
            prompts = kwargs['prompt']

        return self.processor(
            images=kwargs['image'],
            text=prompts,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

    def decode(self, outputs, **kwargs):
        skip_special_tokens = kwargs.get('skip_special_tokens', True)
        clean_up_tokenization_spaces = kwargs.get('clean_up_tokenization_spaces', False)
        return self.processor.batch_decode(
            outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )

    def generate(self, inputs: Dict[str, Any], **kwargs) -> Any:
        # max_len = kwargs.get('max_len', 1000)
        max_new_tokens = kwargs.get('max_new_tokens', 100)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        # Only return newly generated tokens, since the chat template (and
        # thus the "assistant" marker) differs across model backbones.
        return output_ids[:, inputs['input_ids'].shape[1]:]
