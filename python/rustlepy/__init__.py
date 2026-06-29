"""Rustle — Rust-accelerated, librosa-compatible audio feature extraction."""

from importlib.metadata import PackageNotFoundError, version

from ._rustlepy import rms

__all__ = ["rms"]

try:
    __version__ = version("rustlepy")
except PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0.0.0+unknown"
