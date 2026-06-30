# Changelog

All notable changes to this project's memory/quantization work are documented
here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **Post-training quantization** ([src/quantize.py](src/quantize.py)). New module
  that shrinks a trained checkpoint's parameter memory by packing Linear and
  Embedding weights to low-bit integers with per-group scales/biases, via MLX's
  `nn.quantize`.
  - `quantize_model(model, group_size, bits)` — in-place weight quantization.
    `nn.quantize`'s default predicate skips layers whose dimensions are not a
    multiple of `group_size`, so the weight-tied LM head (the token embedding
    reused via `as_linear`) stays correct after quantization.
  - `save_quantized` / `load_quantized` — round-trip a quantized checkpoint,
    storing `group_size`/`bits` in the safetensors metadata so the quantized
    sub-modules can be rebuilt with matching shapes before loading.
  - `quantization_params(path)` — detect whether a checkpoint is quantized.
  - `model_nbytes(model)` — measure resident parameter memory.
- **Transparent quantized loading in generation**
  ([src/generate.py](src/generate.py)). `main()` now loads checkpoints through
  `load_quantized`, which auto-detects quantized files from their metadata and
  falls back to a plain `load_weights` for full-precision checkpoints.

### Measured
- 8-bit / group_size 64 on a 2-layer 128-dim smoke model: parameter memory
  **−71%** with logit cosine similarity **0.99994** vs. full precision — i.e.
  large memory savings with no meaningful accuracy loss. Verified end-to-end
  (quantize → save → reload → identical logits) by `python -m src.quantize`.
