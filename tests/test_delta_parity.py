"""Parity test: rustlepy.delta vs librosa.feature.delta.

librosa.feature.delta == scipy.signal.savgol_filter(data, width,
polyorder=order, deriv=order, mode="interp"). Tested on plain random data
(isolates delta) across widths/orders, plus on a real mel spectrogram.
"""

import numpy as np
import pytest

import rustlepy as rp

librosa = pytest.importorskip("librosa")


@pytest.mark.parametrize("width,order", [(9, 1), (9, 2), (7, 1), (5, 1), (3, 1), (5, 2)])
def test_delta_matches_librosa(width, order):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((24, 200)).astype(np.float64)

    ours = rp.delta(data, width=width, order=order)
    ref = librosa.feature.delta(data, width=width, order=order)

    assert ours.shape == ref.shape
    np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-6)


def test_delta_on_mel():
    rng = np.random.default_rng(2)
    y = rng.standard_normal(22050).astype(np.float64)
    mel = rp.melspectrogram(y, sr=22050)

    ours = rp.delta(mel)
    ref = librosa.feature.delta(mel)

    assert ours.shape == ref.shape
    np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-5)
