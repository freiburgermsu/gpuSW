# Changelog

All notable changes to `gpusw` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-06-11

### Added
- **Apple-Silicon Metal backend** — a second GPU backend that runs the *same* score-only
  affine-gap Gotoh kernel as CUDA, behind the same API. The Metal Shading Language source is
  compiled **at runtime by the OS Metal framework** (the NVRTC analog: no Xcode, no offline
  `metal`/`metallib` toolchain). New modules `gpusw.metal_kernel` (MSL generator, parallel to
  `gpusw.kernel`), `gpusw._metal` (MTLComputePipelineState cache + PyObjC boundary, parallel to
  `gpusw._compile`), and the `gpusw.backends` package (`Backend` contract + `cuda.py` / `metal.py`).
- **Backend selection** on `align_score` / `align_pairs` / `Aligner` via `backend="auto"`
  (prefer CUDA, else Metal), `"cuda"`, or `"metal"`. `Aligner.backend`, `Aligner.metal_source()`,
  `gpusw.metal_available()`, and `gpusw.available_backends()` are new; `gpusw info` and the CLI
  `--backend` flag report/select both backends. The engine in `aligner.py` is now
  device-agnostic; the CUDA path is unchanged (same NVRTC caches, timing, on-device top-k).
- `[metal]` install extra (`pyobjc-framework-Metal` + `libdispatch`, macOS only), macOS classifier.
- **Metal verification:** `tests/test_metal_bitexact.py` and `tests/test_metal_render.py` (the
  twins of the CUDA test files; `@pytest.mark.metal`, auto-skipped without a Metal GPU);
  `examples/metal_16s_align.py` (on-box 16S benchmark + Biopython correctness) and
  `examples/metal_bvbrc_validation.py` (reproduces the EmilyKin study's *precomputed* CPU
  scores on Metal — 100/100 exact, md5-confirmed, no CPU recompute); `docs/FINDINGS_16S_METAL.md`.

### Fixed
- **Metal large-launch correctness.** MSL's `[[thread_position_in_grid]]` is a 32-bit `uint`,
  so a single dispatch with `n_queries × n_refs ≥ 2³²` silently wrapped and left most outputs
  unwritten. The engine now chunks cross products / pair batches so no one launch exceeds the
  backend's `max_threads_per_launch` (2³¹ on Metal); CUDA's 64-bit index is unbounded and its
  default single-launch behavior is unchanged. (Found by the adversarial review.)
- **"query too long" message.** The advertised maximum query length now reports a value whose
  128-aligned `MAXQ` bucket actually fits the per-thread memory budget (it was off by one
  bucket, naming a length that itself raised); fixed for both the Metal and CUDA paths.

### Notes
- Verified on an Apple **M4** (10-core GPU): Metal is integer-identical to the CPU oracle (and
  thus to CUDA) across all modes/matrices/dtypes, reproduces the study's precomputed CPU
  alignment scores exactly (100/100, md5-confirmed), and runs the 3,950-ASV × 343-reference
  exhaustive 16S map at ~15 GCUPS. Full suite: 109 passed (incl. 50 Metal bit-exactness tests).

## [0.1.0] — 2026-06-08

Initial release. Extracted and generalized from the validated 16S rRNA ASV → reference
GPU Smith-Waterman kernel in the `prFBA` project (see `docs/FINDINGS_16S.md`).

### Added
- Runtime-compiled (CuPy/NVRTC, no `nvcc`) affine-gap CUDA kernel, one thread per
  `(query, reference)` pair, score-only.
- Three alignment modes: `local` (Smith-Waterman), `global` (Needleman-Wunsch),
  `semiglobal` (free end gaps via `free_end_gaps_query` / `free_end_gaps_ref`).
- Always-matrix scoring: match/mismatch *or* a full substitution matrix
  (`Matrix.from_match_mismatch` / `named` / `from_ncbi` / `from_array`), with
  bundled `BLOSUM45/50/62/80/90`, `PAM30/70/250`, `EDNAFULL`.
- Arbitrary alphabets and an out-of-alphabet sentinel policy (`mismatch` / `zero` / `error`).
- Automatic int16↔int32 selection with a two-sided overflow envelope; `OverflowRiskError`
  instead of silent wraparound.
- The input funnel (`gpusw.encode.funnel`): lists, dicts, `(id, seq)` iterables,
  `SeqRecord`-likes, FASTA files/`.gz`/text, raw sequences, and pre-encoded buffers.
- High-level `align_score` / `align_pairs`, the reusable `Aligner`
  (`index` / `set_queries` / `score_cross` / `score_pairs` / `top_k` / `gcups` /
  `cuda_source`), and `AlignResult` (`top_k` / `best` / `to_dataframe`).
- Pure-NumPy oracle `cpu_reference_score` / `cpu_reference_matrix`, anchored to Biopython,
  bit-exact with the GPU kernel.
- `gpusw` command-line interface (`cross` / `topk` / `pairs` / `info`).
- Test suite (CPU + GPU), GCUPS benchmark, examples, and CI.
