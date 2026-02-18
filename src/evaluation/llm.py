import os
import asyncio
import aiohttp
from typing import Optional, List, Dict
from deepeval.models.base_model import DeepEvalBaseLLM
from src.generation.llm import OpenRouterLLM
from config import load_config


config = load_config()

class OpenRouterDeepEvalLLM(DeepEvalBaseLLM):
    """
    Wrapper for OpenRouter LLM to be used with DeepEval.
    Supports both synchronous and asynchronous generation via aiohttp.
    """
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str = config["evaluation"]["eval_llm"],
        api_key: Optional[str] = None,
        max_tokens: int = 15000,
        temperature: float = 0.0,
    ):  
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Initialize our internal OpenRouterLLM for sync calls
        self.llm = OpenRouterLLM(
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            config_key="llm"
        )
        print(model)

    def load_model(self):
        """
        DeepEval requires this method. Return the model instance or client.
        """
        return self.llm.client

    def generate(self, prompt: str) -> str:
        """
        DeepEval synchronous generation.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.llm.generate(messages)

    async def a_generate(self, prompt: str) -> str:
        """
        DeepEval asynchronous generation via aiohttp.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.OPENROUTER_API_URL,
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                data = await response.json()

        return data["choices"][0]["message"]["content"]

    def get_model_name(self) -> str:
        return self.model
