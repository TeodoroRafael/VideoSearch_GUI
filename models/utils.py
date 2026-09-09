from transformers import BitsAndBytesConfig


def bitsandbytes_8bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(load_in_8bit=True)
