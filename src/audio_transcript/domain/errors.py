"""Domain and service exceptions."""


class AudioTranscriptError(Exception):
    """Base application error."""


class ConfigurationError(AudioTranscriptError):
    """Raised when required configuration is missing or invalid."""


class ValidationError(AudioTranscriptError):
    """Raised when a request cannot be processed."""


class AudioProcessingError(AudioTranscriptError):
    """Raised when local audio tools fail to inspect or transform audio."""


class ArtifactNotFoundError(AudioTranscriptError):
    """Raised when a stored job artifact cannot be located."""


class StorageError(AudioTranscriptError):
    """Raised when transcript artifacts cannot be written safely."""


class JobNotFoundError(AudioTranscriptError):
    """Raised when a job id does not exist."""


class RetryableProviderError(AudioTranscriptError):
    """Raised for provider failures that are safe to retry."""


class NonRetryableProviderError(AudioTranscriptError):
    """Raised for provider failures that should not be retried."""
