"""
LLM/VLLM Models abstraction layer.
Supports multiple models provider.
"""
import os
import base64
from src.utils.logger import get_logger
from config import load_config
from abc import ABC, abstractmethod
from PIL import Image
from typing import Optional
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv

logger = get_logger(__name__)
load_dotenv()

class VisionModel(ABC):
    """Abstract base class for vision models."""
    @abstractmethod
    def describe_image(self, image: Image.Image, prompt: str) -> str:
        """
        Generate a description of an image.
        
        Args:
            image: PIL Image object
            prompt: Prompt for the vision model
            
        Returns:
            Description text
        """
        pass

    def _image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode()
    

class OpenRouterVision(VisionModel):
    def __init__(
        self,
        model: Optional[str] = None, 
        api_key: Optional[str] = None,
        config_key: str = "vision_llm"):
        """
        Initialize OpenRouter vision model.
        """

        # Get API key from env if not provided
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY in .env file or pass api_key parameter"
            )
        
        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        
        # Load from specified config section
        config = load_config()
        vision_config = config[config_key]
        self.model = model or vision_config["model"]
        
        logger.info(f"Initialized OpenRouter vision model: {self.model} (from {config_key})")

    def describe_image(self, image: Image.Image, prompt: str) -> str:
        """
        Generate a description of an image using OpenRouter.
        
        Args:
            image: PIL Image object
            prompt: Prompt for the vision model
            
        Returns:
            Description text
        """
        # Convert image to base64
        img_base64 = self._image_to_base64(image)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1024,
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error describing image with OpenRouter: {e}")
            raise

def load_vision_model(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    config_key: str = "vision_llm"
) -> VisionModel:
    """
    Factory function to load vision model.
    
    Args:
        model: Model name on OpenRouter (optional, loads from config if not provided)
        api_key: OpenRouter API key (optional, reads from OPENROUTER_API_KEY env var)
        config_key: Config section to load from (default: "vision_llm")
        
    Returns:
        VisionModel instance
    """
    return OpenRouterVision(model=model, api_key=api_key, config_key=config_key)