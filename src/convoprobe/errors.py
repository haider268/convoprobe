"""Exceptions raised by convoprobe."""

from __future__ import annotations


class ConvoprobeError(Exception):
    """Base class for convoprobe errors."""


class TransientError(ConvoprobeError):
    """Retryable failure; the conversation is retried on the next run."""


class AdapterLoadError(ConvoprobeError):
    """An adapter spec could not be loaded."""
