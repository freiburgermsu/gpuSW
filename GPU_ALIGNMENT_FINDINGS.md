# GPU-accelerated alignment scoring — design, test, and findings

**Goal.** Add a GPU tier to the 16S ASV → BV-BRC alignment mapping as a *final accuracy/throughput
enhancement*, evaluating both purpose-built tools (MMseqs2-GPU, CUDASW++4.0 — the CUDASW++4.0 engine
that MMseqs2-GPU, *Nature Methods* 2025, is built on) and a custom kernel.

**Bottom line.** The two named tools are **protein-only on the GPU and cannot accelerate this
nucleotide task**. A **custom CUDA Smith-Waterman kernel** (compiled at runtime via CuPy/NVRTC — no
CUDA toolkit needed) was built, **verified bit-exact against Biopython**, and runs the full-database
*exhaustive* search **7.8× faster than CPU** while **reproducing the CPU result 100%**. This is the
recommended GPU method.

---

## 1. Environment (the constraints that drove every decision)

| | |
|---|---|
| GPU | NVIDIA RTX 5070 Ti, 16 GB, **compute capability sm_120 (Blackwell)** |
| Driver / runtime | 595.71.05 / CUDA 13.2 |
| Toolchain | **No `nvcc`, no CUDA toolkit, no conda** — only `uv`/pip wheels; gcc 13.3 |
| Python GPU | PyTorch 2.11+cu130 works on-GPU; CuPy installed (`cupy-cuda13x` 14.1.1, NVRTC 13.2); Triton present but its gcc shim can't build (no `Python.h` — system venv lacks `python3.12-dev`) |

The "no nvcc" + "brand-new sm_120" combination is what eliminates most off-the-shelf GPU aligners.

## 2. Tool evaluation (fact-checked against primary sources)

A multi-agent research pass (per-tool deep-dive → adversarial verification of the install/modality/
sm_120 claims → synthesis) reached **high-confidence, decisive** conclusions:

### MMseqs2-GPU (soedinglab/MMseqs2, release 18-8cc5c; Kallenborn & Steinegger, *Nat. Methods* 2025)
- **GPU path is PROTEIN/PSSM-only.** The GPU engine is the bundled `libmarv` (= CUDASW++4.0).
  Nucleotide-vs-nucleotide (`--search-type 3`) has **no `--gpu` path and runs on CPU only** —
  confirmed by the README ("GPU-accelerated **protein** sequence and profile searches"), the wiki,
  and the paper (gapless filter + gapped SW-Gotoh **with PSSMs, protein-only**). 16S rRNA is
  non-coding, so translating to protein is invalid. **Decisive blocker — modality, not install.**
- Install itself *is* trivial here (static `mmseqs-linux-gpu.tar.gz`, glibc ≥ 2.29 ✓, driver ≥ 525 ✓,
  no nvcc) — but `--gpu 1` would do nothing for our search, yielding only a CPU nucleotide search.
- (Aside: the released GPU binary is CUDA-12.6 with native archs ≤ sm_90; on sm_120 it would rely on
  `compute_90` PTX-JIT — plausible but unverified. Native sm_120 exists only on untagged master. Moot.)

### CUDASW++4.0 (asbschmidt/CUDASW4)
- **Protein-only** (BLOSUM45/50/62/80; no DNA/match-mismatch/EDNAFULL mode) **and needs nvcc**
  (source + Makefile only, no wheel/binary). Two independent hard blockers → **not usable here;
  design inspiration only** (its DP tiling / length-partitioning / SIMD-in-register packing).

### Others (GASAL2, ADEPT, WFA-GPU, NVBIO, Parabricks)
- All require nvcc/a CUDA toolkit or a container → **not feasible in this no-toolkit env**.

> These tool conclusions come from fact-checked vendor documentation/source, not on-box runs (the
> modality blocker makes a run pointless). The custom kernel below, by contrast, is fully on-box tested.

## 3. Implemented solution — custom NVRTC Smith-Waterman kernel (`gpu_align.py`)

Since no off-the-shelf tool fits, the recommendation (and what was built) is a **custom score-only
local Smith-Waterman CUDA kernel**, compiled **at runtime via CuPy `RawModule`/NVRTC** — which needs
**no `nvcc`** (NVRTC ships in the `cupy-cuda13x` wheel and targets the live GPU's sm_120).

- **Algorithm:** affine-gap (Gotoh) **local** SW, score-only. **One CUDA thread per (ASV, reference)
  pair**; the short ASV (~350–561 bp) is the inner DP buffer to minimize per-thread memory; the long
  reference (~1525 bp) is the outer loop.
- **Scoring scheme identical to the CPU Biopython stage** (match +2, mismatch −3, gap_open −5,
  gap_extend −2, local) → GPU scores are **directly comparable** to the existing pipeline. Bases are
  encoded so equality matches Biopython char-equality exactly.
- **int16 DP buffers** (scores ≪ 32 k for ≤ 561 bp) — halves local-memory traffic, ~**2× throughput**.
- Two entry points: `score_cross(qlist)` (all-references cross product, for exhaustive search) and
  `score_pairs(qi, rj)` (arbitrary pairs, e.g. rescoring an edlib candidate shortlist).
- Note (per research): a score-only kernel does **not** yield % identity for free — % identity /
  coordinates are recovered for the surviving top hits via the existing CPU re-align (cheap).

## 4. Validation & performance (on-box, RTX 5070 Ti)

`python gpu_align.py` → `gpu_align_stats.json`:

| metric | result |
|---|---|
| **Correctness vs Biopython** (400 random real ASV×ref pairs) | **400/400 exact, max abs diff 0** |
| GPU-exhaustive: 20 ASVs × **459,301** refs | **65.9 s @ 57.3 GCUPS** (3.78 T cells) |
| **top-5 md5 overlap vs CPU-exhaustive** | **100.0%** (100/100) |
| **top-1 score match vs CPU-exhaustive** | **20/20** (gpuTop1 == cpuTop1 for every ASV) |
| Speedup vs CPU-exhaustive (511 s, 60 cores) | **7.8×** |
| Extrapolated full 3,950-ASV exhaustive | **~3.6 h GPU** vs **~28 h CPU** |

The GPU kernel reproduces the CPU-exhaustive validation (`asv_top5_alignment_hits_validation.json`)
**exactly** — same optimal scores and the same top hits — confirming both the kernel's correctness and
that GPU SW is a faithful, faster substitute for the CPU SW.

## 5. What this buys the pipeline

- **Accuracy headroom.** The CPU pipeline relies on an edlib top-500 prefilter (validated to lose
  nothing on a 20-ASV stratified sample, but still a heuristic). GPU SW makes **prefilter-free,
  fully-exhaustive** scoring of all 459 k references tractable (~3.6 h for all 3,950 ASVs), removing
  the heuristic entirely for a definitive answer.
- **Drop-in throughput.** Used as a rescorer over the edlib top-500 (~2 M pairs), the same kernel
  replaces the CPU Biopython stage at far higher throughput, freeing the CPU.
- **57 GCUPS on a consumer GPU** with a simple 1-thread/pair kernel is competitive with published
  per-GPU DNA-SW throughput; there is **headroom** (warp-parallel intra-sequence DP, int16×2 SIMD
  packing à la CUDASW++, shared-memory query tiling) to reach the low-hundreds of GCUPS if desired.

## 6. Reproduce

```bash
# venv = ~/Documents/py_venv ; one-time:  uv pip install --python ~/Documents/py_venv/bin/python cupy-cuda13x
python gpu_align.py        # compiles kernel, checks vs Biopython, benchmarks, validates vs CPU-exhaustive
```

Files: `gpu_align.py` (kernel + driver), `gpu_align_stats.json` (the metrics above).
