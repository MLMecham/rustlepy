//! Mel scale, Slaney mel filterbank, and mel spectrogram — librosa-compatible.
//!
//! Constants and algorithm verified against librosa 0.11.0
//! (`core/convert.py`, `filters.py`, `feature/spectral.py`).

use ndarray::{Array1, Array2};

use crate::stft::stft_power;

const F_SP: f64 = 200.0 / 3.0; // mel/Hz slope in the linear region (f_min = 0)
const MIN_LOG_HZ: f64 = 1000.0; // start of the log region

#[inline]
fn min_log_mel() -> f64 {
    MIN_LOG_HZ / F_SP // = 15.0
}
#[inline]
fn logstep() -> f64 {
    (6.4_f64).ln() / 27.0
}

/// Hz -> mel (Slaney by default; HTK when `htk`). Mirrors `librosa.hz_to_mel`.
pub fn hz_to_mel(hz: f64, htk: bool) -> f64 {
    if htk {
        return 2595.0 * (1.0 + hz / 700.0).log10();
    }
    if hz >= MIN_LOG_HZ {
        min_log_mel() + (hz / MIN_LOG_HZ).ln() / logstep()
    } else {
        hz / F_SP
    }
}

/// mel -> Hz. Mirrors `librosa.mel_to_hz`.
pub fn mel_to_hz(mel: f64, htk: bool) -> f64 {
    if htk {
        return 700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0);
    }
    if mel >= min_log_mel() {
        MIN_LOG_HZ * (logstep() * (mel - min_log_mel())).exp()
    } else {
        F_SP * mel
    }
}

/// `n` frequencies spaced evenly on the mel scale between `fmin` and `fmax`.
fn mel_frequencies(n: usize, fmin: f64, fmax: f64, htk: bool) -> Array1<f64> {
    let min_mel = hz_to_mel(fmin, htk);
    let max_mel = hz_to_mel(fmax, htk);
    if n == 1 {
        return Array1::from_vec(vec![mel_to_hz(min_mel, htk)]);
    }
    let step = (max_mel - min_mel) / (n as f64 - 1.0);
    Array1::from_vec(
        (0..n)
            .map(|i| mel_to_hz(min_mel + step * i as f64, htk))
            .collect(),
    )
}

/// FFT bin center frequencies `k * sr / n_fft`, length `1 + n_fft/2`.
fn fft_frequencies(sr: f64, n_fft: usize) -> Array1<f64> {
    let n_freqs = n_fft / 2 + 1;
    Array1::from_vec((0..n_freqs).map(|k| k as f64 * sr / n_fft as f64).collect())
}

/// Slaney mel filterbank, shape `(n_mels, 1 + n_fft/2)`.
/// Mirrors `librosa.filters.mel(..., norm="slaney")` (computed in f64).
pub fn mel_filterbank(
    sr: f64,
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: f64,
    htk: bool,
) -> Array2<f64> {
    let n_freqs = n_fft / 2 + 1;
    let fftfreqs = fft_frequencies(sr, n_fft);
    let mel_f = mel_frequencies(n_mels + 2, fmin, fmax, htk); // len n_mels + 2

    let mut weights = Array2::<f64>::zeros((n_mels, n_freqs));
    for i in 0..n_mels {
        let f_lower = mel_f[i];
        let f_center = mel_f[i + 1];
        let f_upper = mel_f[i + 2];
        let d_lower = f_center - f_lower; // fdiff[i]
        let d_upper = f_upper - f_center; // fdiff[i+1]
        let enorm = 2.0 / (f_upper - f_lower); // Slaney area normalization
        for j in 0..n_freqs {
            let freq = fftfreqs[j];
            let lower = (freq - f_lower) / d_lower;
            let upper = (f_upper - freq) / d_upper;
            let tri = lower.min(upper).max(0.0);
            weights[[i, j]] = tri * enorm;
        }
    }
    weights
}

/// Mel-scaled power spectrogram, shape `(n_mels, n_frames)`.
/// Mirrors `librosa.feature.melspectrogram(y, sr, ..., power=2.0)`:
/// `mel_basis @ |stft|^power`.
#[allow(clippy::too_many_arguments)]
pub fn melspectrogram(
    y: &[f64],
    sr: f64,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    n_mels: usize,
    fmin: f64,
    fmax: f64,
    power: f64,
    center: bool,
    htk: bool,
) -> Array2<f64> {
    let mut spec = stft_power(y, n_fft, hop_length, win_length, center); // |stft|^2
    if (power - 2.0).abs() > 1e-12 {
        // |stft|^power = (|stft|^2)^(power/2)
        spec.mapv_inplace(|v| v.powf(power / 2.0));
    }
    let fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax, htk);
    fb.dot(&spec)
}
