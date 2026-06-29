"""Parity test: rustlepy.melspectrogram vs librosa.feature.melspectrogram.

Mel is the flagship; the f64 Slaney filterbank vs librosa's float32 basis
accounts for the ~1e-7 floor, so rtol=1e-5 is comfortable. Run: `uv run pytest`.
"""

import numpy as np
import pytest

import rustlepy as rp

librosa = pytest.importorskip("librosa")


@pytest.mark.parametrize("sr,n", [(22050, 22050), (44100, 44100)])
def test_mel_matches_librosa_defaults(sr, n):
    rng = np.random.default_rng(0)
    y = rng.standard_normal(n).astype(np.float64)

    ours = rp.melspectrogram(y, sr=sr)
    ref = librosa.feature.melspectrogram(y=y, sr=sr)

    assert ours.shape == ref.shape
    np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-5)


def test_mel_non_default_params():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(44100).astype(np.float64)

    kw = dict(sr=22050, n_fft=1024, hop_length=256, n_mels=64, fmin=50.0, fmax=8000.0)
    ours = rp.melspectrogram(y, **kw)
    ref = librosa.feature.melspectrogram(y=y, **kw)

    assert ours.shape == ref.shape
    np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-5)
