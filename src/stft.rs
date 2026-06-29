//! STFT front-end shared by spectral features (mel now; MFCC/spectral later).

use ndarray::Array2;
use realfft::RealFftPlanner;

/// Periodic ("fftbins") Hann window of length `win_length`, zero-padded and
/// centered into a length-`n_fft` buffer. Matches
/// `scipy.signal.get_window("hann", win_length, fftbins=True)` followed by
/// `librosa.util.pad_center(..., size=n_fft)`.
pub fn hann_window(win_length: usize, n_fft: usize) -> Vec<f64> {
    use std::f64::consts::PI;
    let mut w = vec![0.0_f64; n_fft];
    let left = n_fft.saturating_sub(win_length) / 2;
    for k in 0..win_length.min(n_fft) {
        // periodic Hann (sym=False): 0.5 - 0.5*cos(2*pi*k / win_length)
        w[left + k] = 0.5 - 0.5 * (2.0 * PI * k as f64 / win_length as f64).cos();
    }
    w
}

/// Power spectrogram `|STFT|^2`, shape `(1 + n_fft/2, n_frames)`.
///
/// librosa defaults: `center=True` zero-pads by `n_fft/2` each side
/// (`pad_mode="constant"`), periodic Hann window, unnormalized rFFT.
pub fn stft_power(
    y: &[f64],
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    center: bool,
) -> Array2<f64> {
    let n_freqs = n_fft / 2 + 1;
    let pad = if center { n_fft / 2 } else { 0 };
    let n = y.len();
    let padded_len = n + 2 * pad;
    if hop_length == 0 || padded_len < n_fft {
        return Array2::zeros((n_freqs, 0));
    }

    let mut buf = vec![0.0_f64; padded_len];
    buf[pad..pad + n].copy_from_slice(y);

    let window = hann_window(win_length, n_fft);
    let n_frames = 1 + (padded_len - n_fft) / hop_length;

    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(n_fft);
    let mut frame = r2c.make_input_vec(); // len n_fft (reused, scratch)
    let mut spectrum = r2c.make_output_vec(); // len n_freqs

    let mut out = Array2::<f64>::zeros((n_freqs, n_frames));
    for t in 0..n_frames {
        let start = t * hop_length;
        for i in 0..n_fft {
            frame[i] = buf[start + i] * window[i];
        }
        r2c.process(&mut frame, &mut spectrum)
            .expect("rfft length invariants hold");
        for f in 0..n_freqs {
            let c = spectrum[f];
            out[[f, t]] = c.re * c.re + c.im * c.im; // |z|^2 (power)
        }
    }
    out
}
