# Rustle (`rustlepy`)

> A Rust-accelerated audio **feature-extraction pipeline** with a Python API — think *a faster, corpus-scale librosa core*.

**Status:** 📐 Planning / pre-alpha. This README *is* the plan.

- **Brand / what we call it:** Rustle
- **pip install:** `rustlepy`
- **import:** `rustlepy`
- **Rust crate (internal):** `rustle-core` — compiled into the wheel, *not* published to crates.io

---

## What it is

Rustle decodes any common audio format, computes features in **Rust** (FFT, feature loops, batch parallelism), and hands them back to Python in **two storage forms — chosen by the consumer**:

- **NumPy dense arrays** — in-memory; for deep learning (spectrogram-as-image) and quick scripting.
- **Parquet feature store** — columnar, on-disk; for tabular/classical ML, querying (DuckDB / Polars), and corpus-scale analysis.

It's *preprocessing for any downstream application*: deep learning, classical ML, analysis, or the few built-in detectors. The feature **engine is the asset**; detectors and apps are thin layers on top.

Two faces:
- a **library** — `import rustlepy`
- a thin **CLI** — `rustle features ./audio --out feats.parquet`

## Why Rust (the honest version)

The win is **not** "a faster single FFT" — librosa already calls compiled C for that. The real, defensible wins are:

1. **Batch / parallel** — process a whole folder with real threads (`rayon`), no GIL. This is where Rustle beats librosa + multiprocessing. *(This is the headline.)*
2. **Fused pipeline** — decode → frame → window → FFT → mel → log, all in Rust, never materializing intermediate Python arrays.

The recurring pattern that justifies Rust here: **many tiny numeric kernels over big arrays** (per-frame features, per-band slopes, per-band AR fits) — exactly where a Python loop dies.

Correctness is anchored to **librosa parity** (`np.allclose` within tolerance). That's a first-class deliverable and the trust story.

## Parallelism model: the unit of work

"Batch" really means two distinct things, and conflating them is a mistake:

- **Batch *across* files** — run N files through the no-GIL `rayon` pool. Applies to **every file, any size**. A small file is one whole unit; the only per-file cost is a task dispatch — exactly what a pool is for. **No threshold, this is the cheap headline, it covers all sizes.**
- **Chunk *within* a file** — slice one file into hop-aligned pieces processed in parallel and stitched. This carries real overhead (halo recompute, stitch/reduce, extra dispatch), so it is **gated by a threshold**. Below threshold the file stays one whole unit; chunking a 3-second clip is pure loss.

So the work primitive is **`(source, range)`** — a whole file *or* a slice — but the "or" only triggers for big files. A folder of songs is N whole units; one 3-hour field recording becomes many chunk-units. Same pool, same stitch-per-source machinery.

**When do we chunk a file? Two triggers:**

| Trigger | Condition | Why |
|---|---|---|
| **Latency** | duration > threshold **and** pool has idle cores | use spare cores on a big file; cut tail latency so one huge file doesn't finish last |
| **Memory** | file > per-worker RAM budget | feasibility (out-of-core) — chunk **regardless** of the latency threshold |

Everything below threshold and within the RAM budget is processed **whole**. The decision is ~free: read duration from the container header (or fall back to file bytes on disk) *before* committing to a full decode.

**Picking the threshold.** Of the chunking overheads, the halo recompute is tiny when chunks are seconds-long (an `n_fft/2 ≈ 1024`-sample halo against a 10s chunk is ~0.2% redone frames). The cost that bites is the **fixed per-chunk dispatch + stitch** floor. So: only chunk into pieces large enough that the fixed cost vanishes, and only when you get at least a few of them — a duration threshold of *order tens of seconds*. Treat the exact number as a **benchmarked tunable with a conservative default**, not a magic constant.

**Honest novelty.** The technique is textbook — overlap-save / overlap-add (1960s), Dask's `map_overlap(fn, depth=...)`, Essentia's streaming dataflow, routine overlap-chunking in ASR. We did **not** invent chunked DSP. The product is the **guarantee + integration**: decode-any-format → file-or-slice as one primitive → no-GIL parallel → **output bit-identical to the non-chunked result** → bounded memory, *in one call*. librosa makes you choose `librosa.stream()` (out-of-core but sequential and fiddly) **or** load-it-all-and-DIY-parallel; we make the correct path the easy path.

**The cost we're signing up for** (so it's not hand-waved): the halo is **per-feature**, not a constant.
- **STFT / mel:** halo = `n_fft/2` samples; cut chunks on `hop_length` boundaries so the frame grid aligns.
- **delta / rolling slope:** halo is in **frames** (±window width), not samples — carried as extra context frames per chunk.
- A **stitch/reduce** step reassembles per source, and **each feature gets its own chunked-vs-whole parity test**. That parity test *is* the feature — without it, chunking is a guess.

**v1 scope:** *batch-across-files* (all sizes) and *within-file frame parallelism* land first — no parity risk. *Threshold-gated chunking* is the **same engine** plus the latency trigger and per-feature parity tests; the **memory trigger** rides along since it's cheap and it's the out-of-core justification. *Pool-saturation-aware dynamic chunking* is a later refinement, not v1.

> ⚠️ The GIL is the *current* enabler, not the whole moat: Python 3.13+ free-threading erodes "Rust escapes the GIL" over time. The durable wins are the **fused pipeline** (no intermediate Python arrays) and **parity-guaranteed out-of-core** — both survive a no-GIL Python.

## How it's organized (layers)

```
Layer 4   Trained ML apps  (genre, speaker ID, transcription)   ← NOT us — we feed these
──────────────────────────────────────────────────────────────  ← the line we stop at
Layer 3   Detectors        (onset, tempo, pitch, novelty)        ← a curated few (the showcase)
Layer 2   Temporal feats   (delta, rolling slope, detrend)       ← us
Layer 1   Core features    (mel, MFCC, STFT, RMS, chroma)        ← us (the engine)
Layer 0   Decode any format → samples                            ← us
```

We own Layers 0–2, include a curated few Layer-3 detectors, and **stop before trained ML** — instead making it trivial to feed Rustle's output into sklearn / torch.

## Features

| Feature | Layer | v1 | Notes |
|---|:---:|:---:|---|
| Decode WAV / FLAC / MP3 / OGG → samples | 0 | ✅ | `symphonia`; mono-mix + resample |
| M4A / AAC, streaming decode | 0 | later | feature-gated codecs |
| STFT | 1 | ✅ | shared front-end |
| **Mel spectrogram** | 1 | ✅ | the flagship; `n_mels`, `n_fft`, … params |
| RMS energy | 1 | ✅ | simplest feature — built **first** as the smoke test |
| MFCC | 1 | v1.x | ~free (DCT of log-mel) |
| Spectral centroid / rolloff / bandwidth / flatness, ZCR | 1 | later | cheap per-frame scalars |
| Chroma (stft → cqt → CENS) | 1 | later | harmony / key / cover-song use cases |
| **Delta, Delta-Delta** | 2 | ✅ | the linear-regression flagship (windowed slope) |
| Rolling slope / R², detrend, Savitzky-Golay smoothing, per-file summaries | 2 | later | the time-series toolkit; feeds the feature store |
| **Onset detection** | 3 | ✅ | the one flagship detector (positive-delta / flux peaks) |
| Tempo / beat, pitch / f0, silence / VAD, AR-novelty, LPC / formants | 3 | later | the "cool stuff on top" |

**Standard params mirror librosa** (so parity tests pass and users get expected output):
`sr, n_fft=2048, hop_length=512, win_length=n_fft, window='hann', n_mels=128, fmin=0, fmax=sr/2, power=2.0, top_db=80`.

## Two storage forms

| Form | Best for | How |
|---|---|---|
| **NumPy dense** `(n_mels, T)` | deep learning (spectrogram-as-image), quick scripting | `rp.melspectrogram(...)` → `np.ndarray` |
| **Parquet store** | tabular/classical ML, querying, corpus-scale analysis | `rp.build_feature_store(...)`, or `layout="long"` → `pl.DataFrame` |

Parquet layout notes: default **frame-tidy** (one row per frame; mels as columns) — `~T` rows/file. Optional **fully-long** (one row per `(frame, mel_bin)`) for tidy plotting — `~n_mels × T` rows, use sparingly. Write with **zstd**, ~256k-row row groups, clustered/sorted on the likely filter key. **DuckDB / Polars are readers, not the storage format** — Parquet stays the universal substrate.

## API sketch (illustrative — finalized at the layout step)

```python
import rustlepy as rp

y, sr = rp.load("song.mp3")                 # decode any format → (np.ndarray, sr)
mel   = rp.melspectrogram("song.mp3", n_mels=128)   # (128, T) dB
mfcc  = rp.mfcc("song.mp3")
rms   = rp.rms("song.mp3")                  # (T,)
d     = rp.delta(mel)                        # per-band slope over time
on    = rp.onsets("song.mp3")               # → onset times (s)
res   = rp.ar_residual(rms, order=8)        # forecast residual = novelty curve

df    = rp.melspectrogram("song.mp3", layout="long")   # → pl.DataFrame
rp.build_feature_store("audio/**/*.flac", "feats.parquet",
                       features=["mel", "mfcc", "rms", "delta"], layout="frame")
```

## Applications (vignettes)

### A. Single-feature analysis — mel spectrogram
```python
import rustlepy as rp, matplotlib.pyplot as plt
mel = rp.melspectrogram("song.mp3", n_mels=128)     # (128, T) dB
plt.imshow(mel, origin="lower", aspect="auto"); plt.show()
```

### B. Delta / motion — find where things happen
```python
import numpy as np
mel    = rp.melspectrogram("drums.wav")
dmel   = rp.delta(mel)                        # per-band slope over time
flux   = np.clip(dmel, 0, None).sum(0)        # total positive change / frame
onsets = rp.onsets("drums.wav")               # → times (s); each spike = a hit
print(len(onsets), "hits:", onsets[:8])
```

### C. AR forecasting residual — novelty / section boundaries
```python
import numpy as np
rms      = rp.rms("song.mp3")                 # (T,) loudness curve
surprise = rp.ar_residual(rms, order=8)       # how unpredictable each frame is
bounds   = np.where(surprise > surprise.mean() + 3*surprise.std())[0]
# big residual ≈ structural change: forecast the series, watch where prediction breaks
```

### D. Corpus → Parquet feature store (the batch story)
```python
rp.build_feature_store("audio/**/*.flac", "features.parquet",
                       features=["mel", "mfcc", "rms", "delta"], layout="frame")
import duckdb
duckdb.sql("SELECT file, AVG(rms) FROM 'features.parquet' GROUP BY file ORDER BY 2 DESC")
```

### E. The cool one — a spoken-password door (person + passphrase)

Unlock only when **the right person** says **the right secret phrase** — two independent checks.
Rustle does the **feature prep + the enrollment store**; pretrained models do the smart part.

```python
import rustlepy as rp, numpy as np
from glob import glob
from speaker_model import embed         # external: voice → vector (same person → similar)
from asr_model import transcribe        # external: voice → text (Whisper-style)

PASSPHRASE = "open sesame"

# enrollment (once): store the authorized voiceprint in the Parquet feature store
clips      = glob("alice/*.wav")
mels       = [rp.melspectrogram(c, n_mels=80) for c in clips]   # consistent prep
voiceprint = np.mean([embed(m) for m in mels], axis=0)

# at the door
def unlock(clip):
    m    = rp.melspectrogram(clip, n_mels=80)
    who  = cosine(embed(m), voiceprint) > 0.75   # right PERSON?  (speaker embedding)
    what = fuzzy(transcribe(clip), PASSPHRASE)   # right PHRASE?  (ASR)
    return who and what
```

What makes this honest:
- **"LLM" → split the job by question.** *Identity* ("who?") needs a **speaker-embedding** model, **not** an LLM. The spoken **password** ("what did they say?") is where ASR / an LLM (Whisper + fuzzy text match) belongs. Two different models, two different questions.
- **Rustle's role:** fast, *identical* feature prep for enrollment and verification (mismatched preprocessing is the #1 cause of bad accuracy), plus the enrollment Parquet store and batch processing of clips.
- **Pure-Rustle baseline (no neural net):** MFCC + delta → nearest-neighbor distance is a legitimate weak speaker-ID baseline — nice for showing the features alone do real work before a pretrained model upgrades accuracy.
- ⚠️ **Educational, not production security:** vulnerable to replay (a recording of Alice). Real systems add liveness / anti-spoofing. Frame it as a demo.

*(Later apps in the same spirit: key/chord detection & cover-song matching via chroma; tempo via onset-envelope periodicity.)*

## Tech stack (exact versions pinned at scaffold time)

- **Bridge:** `pyo3` + `numpy` crate + `maturin` (uv-installable wheel)
- **Decode:** `symphonia` (+ optional `hound` WAV fast-path)
- **DSP:** `rustfft`, `ndarray`
- **Parallel:** `rayon` (release the GIL for batch jobs)
- **Columnar:** `polars` + `pyo3-polars` + Arrow (near-zero-copy DataFrame handoff)

> ⚠️ pyo3 / numpy-crate / symphonia move fast and have breaking changes between minor versions. **Pin versions and check current docs at scaffold time** — don't trust pasted snippets (including this README's).

## Planned project layout

```
rustlepy/
├── README.md            ← this plan
├── pyproject.toml       ← build-backend = "maturin"
├── Cargo.toml           ← crate: rustle-core
├── src/
│   ├── lib.rs           ← #[pymodule] rustlepy; wires submodules
│   ├── io.rs            ← decode (symphonia) → samples
│   ├── stft.rs          ← STFT front-end (rustfft)
│   ├── features/        ← mel, mfcc, rms, (later) chroma, spectral
│   ├── temporal.rs      ← delta, (later) rolling/detrend/AR
│   ├── detect.rs        ← onset, (later) tempo/pitch
│   └── store.rs         ← Parquet writer (polars)
├── python/rustlepy/     ← thin Python wrappers, types, CLI
└── tests/               ← librosa parity tests (np.allclose) + benchmarks
```

## v1 = "done" when

1. `mel`, `rms`, `delta` match librosa within tolerance — parity tests green.
2. `uv`-installable via maturin; imports and runs.
3. Onset detector works end-to-end (file → timestamps).
4. Parallel directory → Parquet feature store works.
5. A benchmark showing **batch** speedup vs librosa, with honest methodology.
6. One example notebook (vignettes A–D).

**Build order:** RMS (smoke-test the whole Rust→NumPy→uv loop) → mel → delta → onset → Parquet store → batch/rayon.

## Non-goals (what locks the scope)

- ❌ Trained-model applications in-package (classifiers, transcription) — we *feed* these, with recipes.
- ❌ Audio generation / vocoding (mel → audio).
- ❌ Real-time / streaming (v1).
- ❌ DAW / effects / playback.

## Naming

`rustle` is taken on PyPI (an RLE lib) and crates.io (a speaker-keep-awake tool), so the distribution is **`rustlepy`** and the internal crate is **`rustle-core`** (unpublished, baked into the wheel). We say "Rustle" out loud.

---

## Next steps

- [ ] Lock module/API surface and function signatures (the layout step).
- [ ] Verify + pin current crate versions.
- [ ] Scaffold the buildable skeleton (`pyproject.toml`, `Cargo.toml`, `lib.rs`, a parity test).
- [ ] Implement the RMS smoke-test path end-to-end.
