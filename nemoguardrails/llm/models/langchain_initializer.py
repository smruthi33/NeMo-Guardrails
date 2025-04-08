# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for initializing LangChain models with proper error handling."""

import logging
from importlib.metadata import version
from typing import Any, Callable, Dict, Literal, Optional, Union

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM

from nemoguardrails.llm.providers.providers import (
    _get_chat_completion_provider,
    _get_text_completion_provider,
    _parse_version,
)

log = logging.getLogger(__name__)


class ModelInitializationError(Exception):
    """Raised when model initialization fails."""

    pass


InitStrategy = Callable[
    [str, str, Dict[str, Any]], Optional[Union[BaseChatModel, BaseLLM]]
]


class ModelInitStrategy:
    """A strategy for initializing a model with its supported modes."""

    def __init__(
        self,
        strategy_func: InitStrategy,
        supported_modes: list[Literal["chat", "text"]],
    ):
        self.strategy_func = strategy_func
        self.supported_modes = supported_modes

    def supports_mode(self, mode: Literal["chat", "text"]) -> bool:
        """Check if this strategy supports the given mode."""
        return mode in self.supported_modes

    def execute(
        self, model_name: str, provider_name: str, kwargs: Dict[str, Any]
    ) -> Optional[Union[BaseChatModel, BaseLLM]]:
        """Execute this strategy to initialize a model."""
        return self.strategy_func(
            model_name=model_name, provider_name=provider_name, kwargs=kwargs
        )

    def __str__(self) -> str:
        return f"{self.strategy_func.__name__}(modes={self.supported_modes})"


def try_strategy(
    strategy: ModelInitStrategy,
    model_name: str,
    provider_name: str,
    mode: Literal["chat", "text"],
    kwargs: Dict[str, Any],
):
    """Wrap a strategy execution with a try/except to capture errors."""
    # Skip strategies that don't support the requested mode
    if not strategy.supports_mode(mode):
        log.debug(
            f"Skipping strategy: {strategy.strategy_func.__name__} for model: {model_name} "
            f"and provider: {provider_name} as it doesn't support mode: {mode}"
        )
        return None

    try:
        log.debug(
            f"Trying strategy: {strategy.strategy_func.__name__} for model: {model_name} and provider: {provider_name}"
        )
        result = strategy.execute(
            model_name=model_name, provider_name=provider_name, kwargs=kwargs
        )
        log.debug(f"Strategy {strategy.strategy_func.__name__} returned: {result}")
        if result is not None:
            return result
    except ValueError as e:
        log.error(
            f"ValueError encountered in strategy {strategy} "
            f"for model: {model_name} and provider: {provider_name}: {e}"
        )
    return None


def init_langchain_model(
    model_name: Optional[str],
    provider_name: str,
    mode: Literal["chat", "text"],
    kwargs: Dict[str, Any],
) -> Union[BaseChatModel, BaseLLM]:
    """Initialize a LangChain model using a series of strategies."""
    if not model_name:
        raise ModelInitializationError(
            f"Model name is required for provider {provider_name}"
        )

    # define strategies with their supported modes
    strategies: list[ModelInitStrategy] = [
        ModelInitStrategy(
            _handle_model_edge_cases, ["chat", "text"]
        ),  # special case handlers
        ModelInitStrategy(
            _init_chat_completion_model, ["chat"]
        ),  # preferred -> chat completion
        ModelInitStrategy(
            _init_community_chat_models, ["chat"]
        ),  # fallback -> Community chat models
        ModelInitStrategy(_init_text_completion_model, ["text"]),  # text completion
    ]

    last_exception = None
    for strategy in strategies:
        result = try_strategy(
            strategy=strategy,
            model_name=model_name,
            mode=mode,
            provider_name=provider_name,
            kwargs=kwargs,
        )
        if result is not None:
            return result

    raise ModelInitializationError(
        f"Failed to initialize model {model_name} with provider {provider_name} in {mode} mode"
    ) from last_exception


def _init_chat_completion_model(
    model_name: str, provider_name: str, kwargs: Dict[str, Any]
) -> BaseChatModel:  # noqa #type: ignore
    """Initialize a chat completion model.

    Args:
        model_name: Name of the model to initialize
        provider_name: Name of the provider to use
        kwargs: Additional arguments to pass to the model initialization

    Returns:
        An initialized chat completion model

    Raises:
        ValueError: If the model cannot be initialized as a chat model
    """

    try:
        return init_chat_model(
            model=model_name,
            model_provider=provider_name,
            # configurable_fields="any",
            **kwargs,
        )
    except ValueError as e:
        raise


def _init_text_completion_model(
    model_name: str, provider_name: str, kwargs: Dict[str, Any]
) -> BaseLLM:
    """Initialize a text completion model.

    Args:
        model_name: Name of the model to initialize
        provider_name: Name of the provider to use
        kwargs: Additional arguments to pass to the model initialization

    Returns:
        An initialized text completion model

    Raises:
        RuntimeError: If the provider is not found
    """
    provider_cls = _get_text_completion_provider(provider_name)
    if provider_cls is None:
        raise ValueError()
    kwargs = _update_model_kwargs(provider_cls, model_name, kwargs)
    return provider_cls(**kwargs)


def _init_community_chat_models(
    model_name: str, provider_name: str, kwargs: Dict[str, Any]
) -> BaseChatModel:
    """Initialize community chat models.

    Args:
        provider_name: Name of the provider to use
        model_name: Name of the model to initialize
        kwargs: Additional arguments to pass to the model initialization

    Returns:
        An initialized chat model

    Raises:
        ImportError: If langchain_community is not installed
        ModelInitializationError: If model initialization fails
    """
    provider_cls = _get_chat_completion_provider(provider_name)
    if provider_cls is None:
        raise ValueError()
    kwargs = _update_model_kwargs(provider_cls, model_name, kwargs)
    return provider_cls(**kwargs)


def _init_gpt35_turbo_instruct(
    model_name: str, provider_name: str, kwargs: Dict[str, Any]
) -> BaseLLM:
    """Initialize GPT-3.5 Turbo Instruct model.

    Currently init_chat_model from langchain infers this as a chat model.
    This is a bug in langchain, and we need to handle it here.

    This model requires text completion initialization.

    Args:
        model_name: Name of the model to initialize
        provider_name: Name of the provider to use
        kwargs: Additional arguments to pass to the model initialization

    Returns:
        An initialized text completion model

    Raises:
        ModelInitializationError: If model initialization fails
    """
    try:
        return _init_text_completion_model(
            model_name=model_name,
            provider_name=provider_name,
            kwargs=kwargs,
        )
    except Exception as e:
        raise ModelInitializationError(
            f"Failed to initialize text completion model {model_name}: {str(e)}"
        )


def _init_nvidia_model(model_name: str, provider_name: str, kwargs) -> BaseChatModel:
    """Initialize NVIDIA AI Endpoints model.

    Args:
        model_name: Name of the model to initialize
        provider_name: Name of the provider to use
        **kwargs: Additional arguments to pass to the model initialization

    Returns:
        An initialized chat model

    Raises:
        ImportError: If langchain_nvidia_ai_endpoints is not installed
        ModelInitializationError: If model initialization fails
    """
    try:
        from nemoguardrails.llm.providers._langchain_nvidia_ai_endpoints_patch import (
            ChatNVIDIA,
        )

        package_version = version("langchain_nvidia_ai_endpoints")

        if _parse_version(package_version) < (0, 2, 0):
            raise ValueError(
                "langchain_nvidia_ai_endpoints version must be 0.2.0 or above."
                " Please upgrade it with `pip install langchain-nvidia-ai-endpoints --upgrade`."
            )

        return ChatNVIDIA(model=model_name, **kwargs)
    except ImportError as e:
        raise ImportError(
            "Could not import langchain_nvidia_ai_endpoints, please install it with "
            "`pip install langchain-nvidia-ai-endpoints`."
        )


# special model handlers
_SPECIAL_MODEL_HANDLERS = {
    "gpt-3.5-turbo-instruct": _init_gpt35_turbo_instruct,
}

# provider-specific handlers
_PROVIDER_HANDLERS = {
    "nvidia_ai_endpoints": _init_nvidia_model,
    "nim": _init_nvidia_model,
}


def _handle_model_edge_cases(
    model_name: str, provider_name: str, kwargs: Dict[str, Any]
) -> Optional[Union[BaseChatModel, BaseLLM]]:
    """Handle model initialization for special cases that need custom logic.

    This function handles edge cases where standard initialization methods
    don't work properly. It looks up handlers in the registry and dispatches
    to the appropriate initialization function.

    Args:
        provider_name: Name of the provider to use
        model_name: Name of the model to initialize
        kwargs: Additional arguments to pass to the model initialization

    Returns:
        An initialized model for special cases, or None if no special handler exists
    """
    for pattern, handler in _SPECIAL_MODEL_HANDLERS.items():
        if pattern in model_name:
            return handler(model_name, provider_name, kwargs)

    if provider_name in _PROVIDER_HANDLERS:
        handler = _PROVIDER_HANDLERS[provider_name]
        return handler(model_name, provider_name, kwargs)

    return None


def _update_model_kwargs(provider_cls: type, model_name: str, kwargs: dict) -> Dict:
    """Update kwargs with the model name based on the provider's expected fields.

    If provider_cls.model_fields contains 'model' or 'model_name',
    sets the corresponding key in kwargs to model_name.
    """
    for key in ("model", "model_name"):
        if key in getattr(provider_cls, "model_fields", {}):
            kwargs[key] = model_name
    return kwargs


# def register_model_handler(model_pattern: str, handler: Callable) -> None:
#     """Register a new handler for models matching the given pattern.
#
#     Args:
#         model_pattern: String pattern to match in model names
#         handler: Handler function that accepts (model_name, provider_name, **kwargs)
#     """
#     _SPECIAL_MODEL_HANDLERS[model_pattern] = handler
#
#
# def register_provider_handler(provider_name: str, handler: Callable) -> None:
#     """Register a new handler for a specific provider.
#
#     Args:
#         provider_name: Name of the provider
#         handler: Handler function that accepts (model_name, provider_name, **kwargs)
#     """
#     _PROVIDER_HANDLERS[provider_name] = handler
