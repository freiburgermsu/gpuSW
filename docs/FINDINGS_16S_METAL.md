# Apple-Silicon (Metal) GPU alignment scoring — design, test, and findings

> **Provenance.** This is the Apple-Silicon counterpart of [`FINDINGS_16S.md`](FINDINGS_16S.md)
> (the original NVIDIA/CUDA validation). The Metal backend was developed and verified the same
> way the CUDA tier was — on the same EmilyKin 16S ASV data — but on a consumer MacBook with no
> NVIDIA GPU and no Xcode. File names below refer to the installable `gpusw` package
> (`src/gpusw/metal_kernel.py`, `src/gpusw/_metal.py`, `src/gpusw/backends/metal.py`,
> `examples/metal_16s_align.py`, `examples/metal_bvbrc_validation.py`).

**Goal.** Add a second GPU backend — Apple Silicon via Metal — that performs the *same*
accelerated Smith-Waterman scoring as the CUDA code, behind the same `gpusw` API, and is
**bit-exact** with both the CUDA kernel and the CPU oracle.

**Bottom line.** A custom **Metal Shading Language (MSL)** Smith-Waterman kernel — compiled
**at runtime by the OS Metal framework** (the analog of CUDA's NVRTC; **no Xcode, no offline
`metal`/`metallib` toolchain**) — was built, verified **integer-identical to the CPU oracle**
across all modes/matrices/dtypes, and shown to **reproduce the EmilyKin pipeline's precomputed
CPU (Biopython) alignment scores exactly (100/100, max abs diff 0)** while re-running only the
GPU kernel. On an Apple **M4** it scores the full 3,950-ASV × 343-reference 16S map (1.35 M
pairs) in ~51 s at **~15 GCUPS**.

---

## 1. Environment (the constraints that drove every decision)

| | |
|---|---|
| Machine | MacBook Air, **Apple M4** (4P + 6E CPU, **10-core GPU**), 16 GB unified memory |
| GPU API | **Metal 4**, single default system device, unified memory (`hasUnifiedMemory == True`) |
| OS | macOS 26.5 (build 25F71) |
| Toolchain | **No Xcode, no `metal`/`metallib` CLI** — only Command-Line Tools; Python 3.13 + `uv`/pip wheels |
| Python GPU | PyObjC `pyobjc-framework-Metal` (runtime MSL compile via `newLibraryWithSource:options:error:`); **no CuPy / no NVIDIA GPU** present |

The "no Xcode" + "no offline shader compiler" combination is the Apple mirror of the original
"no `nvcc`, no CUDA toolkit" constraint — and it is what makes a **runtime-compiled** kernel the
right design on both platforms: ship source, compile on the live GPU.

## 2. Why a custom Metal kernel (same reasoning as CUDA, one platform over)

The off-the-shelf GPU aligners surveyed for the CUDA tier (MMseqs2-GPU, CUDASW++4.0, GASAL2,
ADEPT, …) are **CUDA-only** — none run on an Apple GPU at all, independent of the protein-only
and `nvcc` blockers already documented in [`FINDINGS_16S.md`](FINDINGS_16S.md). So on Apple
Silicon there is *no* off-the-shelf option for an exhaustive nucleotide match/mismatch
Smith-Waterman search; a custom kernel is the only path, exactly as it was on the new
`sm_120` NVIDIA card.

## 3. Implemented solution — runtime-compiled MSL Smith-Waterman kernel

A custom **score-only affine-gap (Gotoh) Smith-Waterman/NW** kernel, generated per scheme and
**JIT-compiled at runtime by the Metal framework** — the structural twin of the CUDA generator:

- **One GPU thread per `(query, reference)` pair**; the short ASV (≤ 561 bp) is the inner DP
  buffer (two `MAXQ`-sized per-thread `short`/`int` arrays), the long 16S reference is the outer
  loop — identical to the CUDA kernel.
- **Textually parallel translation.** `src/gpusw/metal_kernel.py` emits the *same* recurrence as
  `src/gpusw/kernel.py`; the only differences are mechanical and inherent to the GPU API:
  `unsigned char`→`uchar`, `device`/`constant` address-space qualifiers, `__global__`→`kernel`
  entry points with `[[buffer(n)]]` bindings and `[[thread_position_in_grid]]`. The mode /
  free-end-gap branches, the int16↔int32 store logic, the always-`SUB[a*NS+b]` substitution
  table, and the empty-sequence edges are character-for-character the same.
- **Two honest API-level differences** (documented in `backends/metal.py`): CUDA's `grid×block`
  launch becomes Metal `dispatchThreadgroups` with a ceil-divided grid and the same in-kernel
  `if (t >= total) return;` bound (identical coverage); and per-query top-k partitions on the
  host-visible unified-memory result (NumPy `argpartition`) rather than via CuPy's on-device
  `argpartition` — cheap precisely because device memory *is* host memory on Apple Silicon.
- **Same scheme as the EmilyKin CPU pipeline** (local SW, match +2 / mismatch −3 / gap_open −5 /
  gap_extend −2 — confirmed against `bvbrc_alignment_hits/edlib_biopython_hits.py`), so Metal
  scores are directly comparable to that pipeline's `align_score` values.
- **int16 DP buffers** for these ≤ 561 bp ASVs (scores ≪ 32 k) — chosen automatically by the
  package's overflow envelope, halving thread-private memory traffic.

The backend is a soft dependency: `import gpusw` and the CPU oracle need no GPU; the Metal path
imports PyObjC lazily and raises an actionable error if it is missing.

## 4. Validation & performance (on-box, Apple M4)

### 4a. Bit-exactness vs the CPU oracle — `tests/test_metal_bitexact.py`

The Metal twin of `tests/test_gpu_bitexact.py`: **integer-identical to the oracle** across
local/global/semiglobal (all four end-gap combinations) × {match-mismatch, EDNAFULL, BLOSUM62}
× {int16, int32, auto-promotion} × edge cases (empty, identical, disjoint, single-character,
16S-scale lengths). **50/50 Metal tests pass on the M4** (the full suite is 107 passed, 52 CUDA
tests skipped for lack of an NVIDIA GPU). Because the oracle is the same ground truth the CUDA
kernel is checked against, this also makes the Metal scores integer-identical to the CUDA scores.

### 4b. Reproducing the precomputed CPU scores — `examples/metal_bvbrc_validation.py`

The strongest check, and the one that needs **no CPU recompute**: the EmilyKin study already
recorded the exhaustive Biopython top-5 BV-BRC hits for 20 validation ASVs (each with an
`align_score`) in `bvbrc_alignment_hits/asv_top5_alignment_hits_validation.json`. Recovering each
hit's reference *sequence* by `feature_id` from the BV-BRC 16S dump (`BV_BRC_16S.json`) and
scoring the exact `(ASV, reference)` pair on Metal:

| metric | result |
|---|---|
| reference sequences recovered from BV-BRC | **100 / 100** |
| recovered sequence **md5-confirmed** as the one the CPU scored | **100 / 100** |
| **Metal score == precomputed CPU `align_score`** | **100 / 100 exact, max abs diff 0** |
| per-ASV top-1 agreement (within recovered hits) | **20 / 20** |

Biopython scores by raw character equality, so ambiguity codes match exactly (the observed
alphabet was `ACGNT`); the comparison uses a match/mismatch scheme over that alphabet to be
apples-to-apples. Only the GPU kernel re-ran — the CPU scores are the study's originals.

### 4c. Exhaustive throughput — `examples/metal_16s_align.py`

Self-contained on local EmilyKin data (queries `asvs.fasta`, references `mag_16s.fasta`):

| metric | result |
|---|---|
| **Correctness vs Biopython** (400 random real ASV×ref pairs) | **400/400 exact, max abs diff 0** |
| GPU-exhaustive: **3,950 ASVs × 343 refs** (1,354,850 pairs) | **50.9 s @ 15.1 GCUPS** (0.766 T cells) |
| top-5 overlap vs Biopython (25-ASV sample) | **100.0 %** |
| top-1 match vs Biopython (25-ASV sample) | **25 / 25** |
| dtype chosen | int16 (auto) |

(See `docs/metal_align_stats_16s.json` and `docs/metal_bvbrc_validation_stats.json` for the
machine-written metrics.)

## 5. What this buys the project

- **A second first-class backend.** The same `align_score` / `Aligner` API now runs on Apple
  Silicon (`backend="metal"`) or NVIDIA (`backend="cuda"`), with `backend="auto"` picking
  whichever GPU is present — so the validated 16S kernel runs on a laptop, not just a CUDA box.
- **Cross-backend equivalence by construction.** Both kernels are checked against the *same*
  oracle, so a result computed on Apple Silicon is integer-identical to one computed on NVIDIA.
- **~15 GCUPS on a fanless M4** with a simple one-thread-per-pair kernel makes the full
  prefilter-free 3,950-ASV exhaustive 16S map a one-minute operation on a laptop; there is
  headroom (SIMD-group-parallel intra-sequence DP, int16×2 packing, threadgroup query tiling).

## 6. Reproduce

```bash
# Apple Silicon, in a venv (Python >= 3.10):
pip install -e '.[metal,test]'        # pyobjc-framework-Metal + libdispatch + pytest/biopython
pytest -m metal                       # 50 Metal bit-exactness tests vs the oracle

# on-box 16S benchmark + Biopython correctness (self-contained EmilyKin data):
python examples/metal_16s_align.py \
    --asvs ../EmilyKin/asvs.fasta --refs ../EmilyKin/mag_16s.fasta

# reproduce the study's precomputed CPU scores on Metal (needs the BV-BRC sequence dump):
python examples/metal_bvbrc_validation.py \
    --seqs ../BV_BRC_16S.json \
    --val  ../EmilyKin/bvbrc_alignment_hits/asv_top5_alignment_hits_validation.json \
    --asvs ../EmilyKin/asvs.fasta
```

Files: `src/gpusw/metal_kernel.py` (MSL generator), `src/gpusw/_metal.py` (runtime compile +
PyObjC boundary), `src/gpusw/backends/metal.py` (launch/dispatch), `examples/metal_16s_align.py`
and `examples/metal_bvbrc_validation.py` (drivers), `docs/metal_align_stats_16s.json` and
`docs/metal_bvbrc_validation_stats.json` (metrics).
