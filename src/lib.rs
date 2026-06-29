//! Rustle core — the Rust extension module (`rustlepy._rustlepy`).
//!
//! Build order: RMS first (smoke test), then the STFT front-end + mel.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

mod mel;
mod stft;

/// Frame-wise RMS energy over a 1-D signal.
///
/// Matches `librosa.feature.rms` defaults: zero-padding by `frame_length / 2`
/// on each side when `center` (librosa's `pad_mode="constant"`), framing by
/// `hop_length`, then `sqrt(mean(frame**2))` per frame.
fn rms_1d(y: &[f64], frame_length: usize, hop_length: usize, center: bool) -> Vec<f64> {
    let pad = if center { frame_length / 2 } else { 0 };
    let n = y.len();
    let padded_len = n + 2 * pad;
    if hop_length == 0 || padded_len < frame_length {
        return Vec::new();
    }

    let mut buf = vec![0.0_f64; padded_len];
    buf[pad..pad + n].copy_from_slice(y);

    let n_frames = 1 + (padded_len - frame_length) / hop_length;
    let inv = 1.0 / frame_length as f64;
    let mut out = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let start = f * hop_length;
        let frame = &buf[start..start + frame_length];
        let sum_sq: f64 = frame.iter().map(|&x| x * x).sum();
        out.push((sum_sq * inv).sqrt());
    }
    out
}

/// Frame-wise root-mean-square (RMS) energy.
///
/// Mirrors ``librosa.feature.rms(y=..., frame_length=2048, hop_length=512,
/// center=True, pad_mode="constant")`` and returns a 1-D float64 array of
/// per-frame RMS values.
#[pyfunction]
#[pyo3(signature = (y, frame_length = 2048, hop_length = 512, center = true))]
fn rms<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    frame_length: usize,
    hop_length: usize,
    center: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // Copy out of the GIL-bound array, then release the GIL for the compute.
    // (pyo3 0.29 renamed `allow_threads` -> `detach`.)
    let y_vec = y.as_array().to_vec();
    let out = py.detach(move || rms_1d(&y_vec, frame_length, hop_length, center));
    Ok(out.into_pyarray(py))
}

/// Mel-scaled power spectrogram.
///
/// Mirrors ``librosa.feature.melspectrogram(y=..., sr=..., n_fft=2048,
/// hop_length=512, n_mels=128, power=2.0, center=True)`` — Slaney mel scale,
/// periodic Hann window, zero padding. Returns a 2-D ``(n_mels, n_frames)``
/// float64 array.
#[pyfunction]
#[pyo3(signature = (y, sr=22050.0, n_fft=2048, hop_length=512, win_length=None, n_mels=128, fmin=0.0, fmax=None, power=2.0, center=true, htk=false))]
#[allow(clippy::too_many_arguments)]
fn melspectrogram<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    sr: f64,
    n_fft: usize,
    hop_length: usize,
    win_length: Option<usize>,
    n_mels: usize,
    fmin: f64,
    fmax: Option<f64>,
    power: f64,
    center: bool,
    htk: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let y_vec = y.as_array().to_vec();
    let win = win_length.unwrap_or(n_fft);
    let fmax = fmax.unwrap_or(sr / 2.0);
    let out = py.detach(move || {
        mel::melspectrogram(
            &y_vec, sr, n_fft, hop_length, win, n_mels, fmin, fmax, power, center, htk,
        )
    });
    Ok(out.into_pyarray(py))
}

#[pymodule]
fn _rustlepy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rms, m)?)?;
    m.add_function(wrap_pyfunction!(melspectrogram, m)?)?;
    Ok(())
}
