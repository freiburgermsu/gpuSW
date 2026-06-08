# gpusw

**GPU-accelerated Smith-Waterman / Needleman-Wunsch alignment *scoring*, with a custom CUDA kernel compiled at runtime — no `nvcc`, no CUDA toolkit, no conda.**

`gpusw` scores one `(query, reference)` sequence pair per CUDA thread using affine-gap (Gotoh) dynamic programming. The kernel is generated and JIT-compiled on your live GPU through CuPy/NVRTC, so the only thing you install is a Python wheel. It is generalized over the **scoring scheme** (match/mismatch *or* a full substitution matrix), the **alignment mode** (local / global / semiglobal), the **alphabet**, and the integer **dtype** — and every GPU result is **bit-exact** with an included pure-NumPy reference implementation (which itself is validated against Biopython).

```python
import gpusw

res = gpusw.align_score(["ACGTACGTACGT", "TTTTGGGG"],
                        ["ACGTACGTACGT", "ACGTAAAA", "GGGGTTTT"],
                        scheme="dna")
res.scores           # (2, 3) int32 cross-product of optimal alignment scores
res.top_k(1)         # [('q0', [('r0', 24)]), ('q1', [('r2', 8)])]
```

Origin: `gpusw` was extracted and generalized from a validated 16S rRNA ASV → reference
alignment kernel (see [`docs/FINDINGS_16S.md`](docs/FINDINGS_16S.md)) that ran the full-database
exhaustive Smith-Waterman search **7.8× faster than a 60-core CPU** while reproducing its
result 100%.

---

## Why

Off-the-shelf GPU aligners (MMseqs2-GPU, CUDASW++4.0, GASAL2, ADEPT, …) are either
**protein-only on the GPU**, or require **`nvcc` / a full CUDA toolkit / a container** to
build. On a fresh consumer GPU (e.g. Blackwell `sm_120`) with only pip/uv wheels, none of
them accelerate a nucleotide match/mismatch search. `gpusw` sidesteps all of that: NVRTC
ships inside the `cupy-cudaXXx` wheel and targets your running GPU's architecture, so a
plain `pip install` is enough.

- **No toolchain.** Pure-Python wheel; the CUDA kernel is compiled at runtime on the GPU you have.
- **Correct by construction.** GPU scores are integer-identical to the bundled CPU oracle, which is anchored to Biopython for local, global and semiglobal modes.
- **General.** DNA/RNA/protein/custom alphabets, any affine scoring scheme, any of three modes, automatic int16↔int32 overflow handling.
- **Fast.** ~70–95 GCUPS on a single RTX 5070 Ti with a simple one-thread-per-pair kernel (see [Benchmarks](#benchmarks)).
- **GPU-optional.** `import gpusw` and the CPU oracle work with no GPU at all.

## Install

```bash
pip install gpusw                 # core (numpy only)

# add the CuPy wheel matching your CUDA major version:
pip install "gpusw[cuda13]"       # CUDA 13.x
pip install "gpusw[cuda12]"       # CUDA 12.x
pip install "gpusw[cuda11]"       # CUDA 11.x

pip install "gpusw[pandas]"       # AlignResult.to_dataframe()
```

If you already manage CuPy yourself, plain `pip install gpusw` is all you need — CuPy is a
soft dependency, imported lazily only on the GPU code paths.

Requirements: Python ≥ 3.10, NumPy ≥ 1.23, and (for GPU use) CuPy + an NVIDIA GPU/driver.

## Quick start

### One call: all-pairs cross product

```python
import gpusw

queries = {"asv1": "ACGTACGTACGT", "asv2": "TTTTGGGG"}
refs    = {"ref1": "ACGTACGTACGT", "ref2": "ACGTAAAA", "ref3": "GGGGTTTT"}

res = gpusw.align_score(queries, refs, scheme="dna")
res.scores              # np.int32 array, shape (n_queries, n_refs)
res.top_k(2)            # per query: [(ref_id, score), ...] best-first
res.best()              # per query: (query_id, best_ref_id, score)
res.to_dataframe()      # long-format pandas frame (needs [pandas])
```

### Reuse: index references once, score many query batches

```python
from gpusw import Aligner

al = Aligner("blosum62").index(protein_db_fasta)   # encode + upload once
for batch in query_batches:
    hits = al.top_k(batch, k=5)                    # GPU argpartition, memory-bounded
    print(al.gcups, "GCUPS")
```

### Paired scoring (1:1 or arbitrary pairs)

```python
gpusw.align_pairs(queries, refs)                       # zip 1:1 (len must match)
gpusw.align_pairs(queries, refs, pairs=[(0, 5), (0, 9), (3, 5)])
```

### No GPU? Score and verify on the CPU

```python
from gpusw import cpu_reference_score, schemes
cpu_reference_score("ACGTACGT", "ACGT", schemes.DNA)   # -> 8  (pure NumPy, no CuPy)
```

### Command line

```bash
gpusw info                                             # GPU + bundled matrices
gpusw cross --queries q.fasta --refs r.fasta --scheme dna           -o scores.csv
gpusw topk  --queries q.fasta --refs r.fasta --scheme blosum62 -k 5 -o hits.csv
gpusw pairs --queries q.fasta --refs r.fasta --scheme dna          -o pairs.csv
```

## Funneling experimental inputs into discrete forms

Every entry point accepts sequences in whatever shape your experiment produced; `gpusw`
funnels them into the single discrete form the kernel needs (a concatenated `uint8` code
buffer + an `int32` offset array). Accepted forms:

| Input | Identifiers |
|---|---|
| `list`/`tuple`/`np.ndarray` of `str` | `q0, q1, …` |
| `dict[name -> seq]` | the keys |
| iterable of `(id, seq)` pairs | the ids |
| Biopython `SeqRecord`s (or anything with `.id`/`.seq`) | `record.id` (no Biopython import needed) |
| a FASTA file path (`.fasta`, `.fa`, `.gz`) | FASTA headers |
| FASTA *text* (a string starting with `>`) | FASTA headers |
| a single raw `str`/`bytes` sequence | `q0` |
| a pre-built `(codes, offsets)` tuple | fast path, no re-encoding |

Out-of-alphabet symbols are mapped to a sentinel code and scored by the `unknown` policy
(`"mismatch"`, `"zero"`, or `"error"`), so messy real-world input never silently corrupts a score.

## Parameter breadth (`Scheme`)

A `Scheme` is the single frozen, hashable object that captures every scoring knob. Pass a
preset name, a `Scheme`, or override individual fields on the helpers.

| Parameter | Default | Meaning |
|---|---|---|
| `mode` | `"local"` | `local` (Smith-Waterman), `global` (Needleman-Wunsch), `semiglobal` |
| `match` / `mismatch` | `2` / `-3` | match/mismatch path (ignored when `matrix` is set) |
| `gap_open` | `-5` | cost of the **first** gap residue (Biopython `open_gap_score`) |
| `gap_extend` | `-2` | cost of each **subsequent** gap residue |
| `matrix` | `None` | a `Matrix` (BLOSUM/PAM/EDNAFULL/custom); overrides match/mismatch |
| `alphabet` | `"ACGT"` | residue order for the match/mismatch path |
| `free_end_gaps_query` / `free_end_gaps_ref` | `True` / `True` | semiglobal: which sequence's terminal gaps are free |
| `unknown` | `"mismatch"` | out-of-alphabet policy: `mismatch` / `zero` / `error` |
| `case_insensitive` | `True` | fold case before encoding |
| `dtype` | `"auto"` | DP/output type: `auto` (promote int16→int32 when a score could overflow) / `int16` / `int32` |

```python
from gpusw import Scheme, Matrix, align_score

# fully custom DNA semiglobal scheme
sch = Scheme(mode="semiglobal", match=5, mismatch=-4, gap_open=-10, gap_extend=-1,
             free_end_gaps_query=True, free_end_gaps_ref=False)

# protein with an explicit matrix
sch = Scheme(mode="local", matrix=Matrix.named("BLOSUM80"), gap_open=-11, gap_extend=-1)

# bring your own matrix (NCBI text format, an array, or match/mismatch)
m = Matrix.from_array("ACGU", [[3,-2,-2,-2],[-2,3,-2,-2],[-2,-2,3,-2],[-2,-2,-2,3]])
align_score(rna_queries, rna_refs, matrix=m, mode="global")
```

**Presets** (`gpusw.schemes`): `DNA`, `DNA_GLOBAL`, `DNA_SEMIGLOBAL`, `BLASTN`, `EDNAFULL`,
`BLOSUM62`, `BLOSUM62_GLOBAL`, `PROTEIN_SEMIGLOBAL`.
**Bundled matrices**: `BLOSUM45/50/62/80/90`, `PAM30/70/250`, `EDNAFULL` — list them with
`gpusw.available_matrices()`.

## How it works

- **Algorithm.** Affine-gap (Gotoh) DP, score-only, one CUDA thread per `(query, reference)` pair. The query is the inner DP buffer (two `MAXQ`-sized per-thread arrays); the reference is the outer loop.
- **Always-matrix scoring.** Even match/mismatch schemes compile to a tiny `NS×NS` substitution table indexed as `s = SUB[a*NS + b]`. This unifies DNA/protein/custom scoring and makes out-of-alphabet handling exact (a sentinel row/col), with no spurious "unknown == unknown" matches.
- **Compile-time specialization.** Mode, gap costs, alphabet size, free-end-gap flags, `MAXQ`, and the DP element type are baked into the generated source as `#define`/`typedef`, so each scheme gets its own branch-free kernel. Compiled modules are cached process-wide keyed on every source-affecting field **plus the GPU architecture**.
- **int16 by default, int32 when needed.** DP arithmetic runs in 32-bit registers; only the stored cells narrow. A two-sided overflow envelope (derived from the score extremes and sequence lengths) picks int16 (≈2× the memory throughput) when safe and promotes to int32 otherwise. Forcing `dtype="int16"` past the safe range raises `OverflowRiskError` rather than wrapping silently.
- **Score-only.** A score-only kernel does not recover percent identity / alignment coordinates; recover those for the surviving top hits with a cheap CPU re-align (e.g. Biopython) — see [`examples/`](examples).

Inspect the generated CUDA for any aligner with `Aligner(...).cuda_source()`.

## Correctness

The package ships its own ground truth: `gpusw.cpu_reference_score` / `cpu_reference_matrix`
is a pure-NumPy affine-Gotoh implementation using the **same** recurrence, gap convention,
substitution-table indexing and free-end-gap handling as the kernel.

- The oracle is anchored to **Biopython `PairwiseAligner`** (200/200 random pairs per mode, including all four semiglobal end-gap combinations).
- The GPU kernel is asserted **integer-identical to the oracle** across modes × scoring schemes × dtypes × edge cases (empty, identical, disjoint, single-character, length-at-`MAXQ`).
- CI runs the no-GPU tests (oracle, encoding, cache-key, kernel rendering) on every push; the GPU bit-exactness tests run wherever a GPU is present.

```bash
pip install "gpusw[test]"
pytest -m "not gpu"     # CPU-only: oracle, encoding, scheme/dtype, rendering
pytest                  # everything, including @gpu bit-exactness (needs a GPU)
```

## Benchmarks

Single **RTX 5070 Ti** (Blackwell `sm_120`), exhaustive cross-product, via
[`benchmarks/bench_gcups.py`](benchmarks/bench_gcups.py):

| Workload | Cells | Throughput |
|---|---|---|
| DNA local, 64×~350 bp queries vs 4000×~1500 bp refs | 0.13 T | **~77 GCUPS** |
| Protein BLOSUM62 local, 48×~300 aa vs 3000×~400 aa | 0.018 T | **~95 GCUPS** |

```bash
python benchmarks/bench_gcups.py --queries 64 --qlen 350 --refs 4000 --rlen 1500
```

There is headroom (warp-parallel intra-sequence DP, int16×2 SIMD packing, shared-memory
query tiling) to push into the low hundreds of GCUPS; the current kernel favors simplicity
and exactness.

## Project layout

```
gpuSW/
├── src/gpusw/
│   ├── __init__.py        # align_score / align_pairs + public re-exports
│   ├── scheme.py          # Scheme: all scoring parameters, dtype/overflow logic
│   ├── matrix.py          # Matrix: match/mismatch, named, from_ncbi, custom
│   ├── schemes.py         # presets (DNA, BLOSUM62, EDNAFULL, ...)
│   ├── encode.py          # the input funnel -> discrete (codes, offsets)
│   ├── kernel.py          # CUDA source generator (per scheme)
│   ├── _compile.py        # NVRTC RawModule cache + lazy CuPy boundary
│   ├── aligner.py         # Aligner engine: score_cross / score_pairs / top_k
│   ├── reference.py       # pure-NumPy oracle (the bit-exactness ground truth)
│   ├── result.py          # AlignResult (top_k, to_dataframe)
│   ├── cli.py             # `gpusw` command line
│   └── data/matrices/     # bundled BLOSUM/PAM/EDNAFULL tables
├── tests/                 # oracle-vs-Biopython, GPU-vs-oracle, encoding, packaging
├── examples/              # worked examples (incl. the original 16S driver)
├── benchmarks/            # GCUPS benchmark
└── docs/                  # design notes + the 16S validation findings
```

## Publishing (maintainers)

```bash
python -m build                       # -> dist/gpusw-<v>-py3-none-any.whl + .tar.gz
python -m twine check dist/*          # metadata + README render
python -m twine upload dist/*         # PyPI (token in ~/.pypirc or TWINE_* env)
```

## License

MIT © Andrew Freiburger. See [LICENSE](LICENSE).
