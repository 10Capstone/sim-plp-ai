class ServiceUnavailableError(RuntimeError):
    """A retrieval dependency is temporarily unavailable."""


class LLMUnavailableError(ServiceUnavailableError):
    """The answer generation dependency is temporarily unavailable."""
