"""
LLM abstraction layer for text generation.
Supports multiple providers via OpenAI-compatible API.
"""
import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from openai import OpenAI
from dotenv import load_dotenv

from src.utils.logger import get_logger
from config import load_config

config = load_config()
logger = get_logger(__name__)
load_dotenv()


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a completion from a list of messages.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
            **kwargs: Optional overrides (max_tokens, temperature)

        Returns:
            Generated text string
        """
        pass

    def generate_stream(self, messages: List[Dict[str, str]], token_callback=None, **kwargs) -> str:
        """
        Stream a completion, calling token_callback for each token chunk.

        Default implementation falls back to generate() and chunks the output.
        Override in subclasses for true streaming.

        Args:
            messages: Chat messages
            token_callback: Callable[[str], None] invoked for each token
            **kwargs: Optional overrides forwarded to generate()

        Returns:
            Full generated text
        """
        text = self.generate(messages, **kwargs)
        if token_callback:
            for char in text:
                token_callback(char)
        return text

    def generate_json(self, messages: List[Dict[str, str]], **kwargs) -> Optional[dict]:
        """
        Generate and parse JSON output from LLM.
        Handles cases where the model wraps JSON in markdown code fences.

        Returns:
            Parsed dict, or None if parsing fails
        """
        raw = self.generate(messages, **kwargs)
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        """Extract and parse JSON from LLM output."""
        cleaned = text.strip()

        # Strip markdown code fences
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: find first { ... } block
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Failed to parse JSON from LLM output: {text[:200]}...")
            return None


class OpenRouterLLM(BaseLLM):
    """OpenRouter LLM using OpenAI-compatible API."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        config_key: str = "llm",
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY in .env file or pass api_key parameter"
            )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        
        # Load from specified config section
        llm_config = config[config_key]
        self.model = model or llm_config["model"]
        self.max_tokens = max_tokens or llm_config["max_tokens"]
        self.temperature = temperature if temperature is not None else llm_config["temperature"]

        logger.info(f"Initialized OpenRouter LLM: {self.model} (from {config_key})")

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate completion via OpenRouter."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        content = response.choices[0].message.content
        logger.debug(f"LLM generated {len(content)} chars")
        return content

    def generate_stream(self, messages: List[Dict[str, str]], token_callback=None, **kwargs) -> str:
        """Generate completion via OpenRouter with streaming enabled.

        Calls token_callback(chunk) for every token chunk received.
        Returns the full concatenated text.
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            stream=True,
        )
        full_content = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                full_content += delta
                if token_callback:
                    token_callback(delta)
        logger.debug(f"LLM streamed {len(full_content)} chars")
        return full_content

    def web_search(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Perform web search via OpenRouter."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            extra_body={
            "plugins": [
                {
                    "id": "web",
                    "max_results": 5,  
                    "search_prompt": "Use these web results to answer the question:"
                }
            ]
        }
        )
        content = response.choices[0].message.content
        logger.debug(f"LLM generated {len(content)} chars")
        return content



def get_llm(provider: str = None, config_key: str = "llm", **kwargs) -> BaseLLM:
    """
    Factory function to create an LLM instance.

    Args:
        provider: Provider name ("openrouter", "openai", "ollama")
        config_key: Config section to load from ("llm", "generation_llm", etc.)
        **kwargs: Override any constructor parameter

    Returns:
        BaseLLM instance
    """
    llm_config = config[config_key]
    provider = provider or llm_config["provider"]
    logger.info(f"Creating LLM: provider={provider}, config={config_key}")

    if provider == "openrouter":
        return OpenRouterLLM(config_key=config_key, **kwargs)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Supported: openrouter"
        )
