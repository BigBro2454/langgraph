"""
Model Factory for LangGraph & LangChain Agents.

Provides resilient model initialization with automatic fallback from cloud providers
(Google Gemini / OpenAI) to local offline models (Ollama qwen2.5:7b).
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


def get_model(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    enable_fallback: bool = True,
) -> BaseChatModel:
    """
    Returns a configured chat model with optional local fallback.

    Priority:
    1. Explicit model_name or Google Gemini if GOOGLE_API_KEY is set.
    2. OpenAI if OPENAI_API_KEY is set and requested.
    3. Fallback to local Ollama (qwen2.5:7b) if cloud model fails or rate limits.
    """
    google_key = os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model_name = os.environ.get("OLLAMA_DEFAULT_MODEL", "qwen2.5:7b")

    primary_model: Optional[BaseChatModel] = None

    # Determine primary model
    if model_name and ("gpt" in model_name.lower()) and openai_key:
        from langchain_openai import ChatOpenAI
        primary_model = ChatOpenAI(model=model_name, temperature=temperature)
    elif google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        target_model = model_name if model_name else "gemini-2.5-flash"
        primary_model = ChatGoogleGenerativeAI(
            model=target_model,
            temperature=temperature,
            convert_system_message_to_human=False,
        )
    elif openai_key:
        from langchain_openai import ChatOpenAI
        primary_model = ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            temperature=temperature,
        )
    else:
        # No cloud API keys; use local Ollama directly
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=ollama_model_name,
            base_url=ollama_base_url,
            temperature=temperature,
        )

    # Attach local Ollama fallback if enabled
    if enable_fallback:
        try:
            from langchain_ollama import ChatOllama
            fallback = ChatOllama(
                model=ollama_model_name,
                base_url=ollama_base_url,
                temperature=temperature,
            )
            return primary_model.with_fallbacks([fallback])
        except Exception as e:
            logger.warning(f"Could not configure Ollama fallback: {e}")
            return primary_model

    return primary_model
