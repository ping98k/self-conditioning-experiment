"""
LLM client module for the Long Horizon Execution Benchmark.

Provides an OpenAI-compatible client for communicating with a
llama.cpp server, supporting multi-turn chat conversations with
configurable generation parameters.
"""

from openai import OpenAI

from config import ModelConfig


class LLMClient:
    """Client for OpenAI-compatible LLM API (llama.cpp server)."""

    def __init__(self, config: ModelConfig):
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        self.model = config.model
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.max_tokens = config.max_tokens

    def chat(self, messages: list[dict]) -> str:
        """Send messages and return the assistant's response text."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""
