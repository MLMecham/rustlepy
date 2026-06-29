# Contributing to Rustle (`rustlepy`)

Mixed Rust + Python project built with [maturin](https://www.maturin.rs/) and
managed with [uv](https://docs.astral.sh/uv/). Rust lives in `src/`, the Python
package in `python/rustlepy/`, tests in `tests/`.

## Dev setup

```bash
uv venv                          # create the dev virtualenv
uv pip install -e ".[dev]"       # build the Rust extension (editable) + dev deps
uv run pytest                    # run the librosa parity tests
```

`maturin` does not need to be installed globally — `uv` fetches it in an
isolated build environment from `[build-system]`, and it's also in the `dev`
extra for the rebuild command below.

## The edit/build loop

- **Editing Python** (`python/rustlepy/*.py`): changes are live (editable install) — just re-run tests.
- **Editing Rust** (`src/*.rs`): the compiled extension must be rebuilt:

  ```bash
  uv run maturin develop --uv
  ```

## Parity is the contract

Every feature ships with a `tests/test_*_parity.py` that asserts
`np.allclose` against the librosa equivalent. A feature without a green parity
test is not done.
