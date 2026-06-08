# Changelog

All notable changes to `gpusw` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

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
