from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class VLMWrapperRetrieval(ABC):
    """
    Abstract base class for a Visual Language Model (VLM) wrapper that supports retrieval tasks.
    """

    model: Optional[Any] = field(default=None)  # The underlying VLM model instance.
    processor: Optional[Any] = field(default=None)  # The processor for handling input data (e.g., images, text).

    def __post_init__(self):
        """
        Post-initialization to ensure that the model and processor are set up correctly.
        """
        if self.model is None:
            raise ValueError("Model must be provided.")
        if self.processor is None:
            raise ValueError("Processor must be provided.")

    @abstractmethod  # Abstract method to process inputs (e.g., images, text) before passing them to the model. 
    # Must be implemented by subclasses.
    def process_inputs(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_embeddings(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_text_embeddings(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_image_embeddings(self, *args, **kwargs):
        pass


@dataclass
class VLMWrapperCaptioning(ABC):
    """
    Abstract base class for a Visual Language Model (VLM) wrapper that supports captioning tasks.
    """

    model: Optional[Any] = field(default=None)  # The underlying VLM model instance.
    processor: Optional[Any] = field(default=None)  # The processor for handling input data (e.g., images, text).

    def __post_init__(self):
        """
        Post-initialization to ensure that the model and processor are set up correctly.
        """
        if self.model is None:
            raise ValueError("Model must be provided.")
        if self.processor is None:
            raise ValueError("Processor must be provided.")

    @abstractmethod
    def process_inputs(self, *args, **kwargs):
        pass

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

    @abstractmethod
    def decode(self, *args, **kwargs):
        pass