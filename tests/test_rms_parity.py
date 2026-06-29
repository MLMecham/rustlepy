"""Parity test: rustlepy.rms must match librosa.feature.rms within tolerance.

This is the v1 trust story in miniature — the same np.allclose check every
feature will get. Run with: `uv run pytest`.
"""

import numpy as np
import pytest

import rustlepy as rp

librosa = pytest.importorskip("librosa")


@pytest.mark.parametrize("n", [22050, 16000, 5000, 2049])
def test_rms_matches_librosa(n):
    rng = np.random.default_rng(0)
    y = rng.standard_normal(n).astype(np.float64)
    frame_length, hop_length = 2048, 512

    ours = rp.rms(y, frame_length=frame_length, hop_length=hop_length)
    ref = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    assert ours.shape == ref.shape
    np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-8)


def test_rms_non_default_params():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(48000).astype(np.float64)
    frame_length, hop_length = 1024, 256

    ours = rp.rms(y, frame_length=frame_length, hop_length=hop_length)
    ref = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-8)
