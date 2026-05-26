from verifiers.clients.anthropic_messages_client import AnthropicMessagesClient
from verifiers.clients.client import Client
from verifiers.clients.nemorl_chat_completions_client import (
    NeMoRLChatCompletionsClient,
)
from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.clients.openai_chat_completions_token_client import (
    OpenAIChatCompletionsTokenClient,
)
from verifiers.clients.openai_completions_client import OpenAICompletionsClient
from verifiers.clients.openai_responses_client import OpenAIResponsesClient
from verifiers.types import ClientConfig


def _load_renderer_client():
    try:
        from verifiers.clients.renderer_client import RendererClient
    except ModuleNotFoundError as e:
        missing = e.name or ""
        if missing == "renderers" or missing.startswith("renderers."):
            raise ImportError(
                "RendererClient requires the renderers extra; install "
                "`verifiers[renderers]`."
            ) from e
        raise

    return RendererClient


def resolve_client(client_or_config: Client | ClientConfig) -> Client:
    """Resolves a client or client config to a client."""
    if isinstance(client_or_config, Client):
        client = client_or_config
        return client
    elif isinstance(client_or_config, ClientConfig):
        client_type = client_or_config.client_type
        match client_type:
            case "openai_completions":
                return OpenAICompletionsClient(client_or_config)
            case "openai_chat_completions":
                return OpenAIChatCompletionsClient(client_or_config)
            case "openai_chat_completions_token":
                return OpenAIChatCompletionsTokenClient(client_or_config)
            case "openai_responses":
                return OpenAIResponsesClient(client_or_config)
            case "renderer":
                RendererClient = _load_renderer_client()
                return RendererClient(client_or_config)
            case "anthropic_messages":
                return AnthropicMessagesClient(client_or_config)
            case "nemorl_chat_completions":
                return NeMoRLChatCompletionsClient(client_or_config)
            case "custom":
                return _resolve_custom_client(client_or_config)
    else:
        raise ValueError(f"Unsupported client type: {type(client_or_config)}")


def _resolve_custom_client(config: ClientConfig) -> Client:
    """Instantiate a user-supplied ``Client`` subclass from ``config.class_path``.

    Lets external packages plug a backend in without modifying verifiers.
    """
    import importlib

    if not config.class_path:
        raise ValueError("ClientConfig with client_type='custom' requires class_path to be set.")
    module_path, _, class_name = config.class_path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"class_path must be a dotted path like 'package.module.ClassName', got {config.class_path!r}"
        )
    cls = getattr(importlib.import_module(module_path), class_name)
    if not (isinstance(cls, type) and issubclass(cls, Client)):
        raise TypeError(
            f"class_path={config.class_path!r} resolved to {cls!r}, "
            f"which is not a verifiers.clients.Client subclass."
        )
    return cls(config)


def __getattr__(name: str):
    if name == "RendererClient":
        return _load_renderer_client()
    raise AttributeError(f"module 'verifiers.clients' has no attribute '{name}'")


__all__ = [
    "AnthropicMessagesClient",
    "NeMoRLChatCompletionsClient",
    "OpenAICompletionsClient",
    "OpenAIChatCompletionsClient",
    "OpenAIChatCompletionsTokenClient",
    "OpenAIResponsesClient",
    "RendererClient",
    "Client",
]
