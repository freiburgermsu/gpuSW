#!/usr/bin/env python
"""
gpu_align.py — GPU-accelerated Smith-Waterman local-alignment SCORING of 16S ASVs
against the BV-BRC reference set, via a custom CUDA kernel compiled at runtime with
CuPy/NVRTC (no nvcc, no CUDA toolkit; runs on the RTX 5070 Ti / sm_120).

One CUDA thread scores one (ASV, reference) pair with affine-gap local SW, using the
SAME scoring scheme as the Biopython stage (match +2, mismatch -3, gap_open -5,
gap_extend -2, local) so GPU scores are directly comparable to the CPU pipeline.

This module is the "final method" GPU tier: it can (a) rescore the edlib candidate
shortlist far faster than CPU Biopython, and (b) make EXHAUSTIVE all-reference SW
scoring tractable (the CPU-exhaustive validation of 20 ASVs took 511 s; full 3950-ASV
exhaustive was ~28 h extrapolated on CPU).

Run directly to: compile the kernel, verify scores against Biopython, benchmark GCUPS,
run GPU-exhaustive SW over all unique refs for the 20 validation ASVs, and confirm it
reproduces the CPU-exhaustive top-5.

Interpreter: ~/Documents/py_venv/bin/python   (needs cupy-cuda13x)
"""
from __future__ import annotations
import json, os, time
import numpy as np
import cupy as cp

import edlib_biopython_hits as P   # reuse loaders + scoring constants

MATCH, MISMATCH, GAP_OPEN, GAP_EXT = int(P.MATCH), int(P.MISMATCH), int(P.GAP_OPEN), int(P.GAP_EXTEND)
MAXQ = 600          # max ASV (query) length; buffers are sized to this (longest ASV = 561)
NEG = -1 << 20

# base encoding: A/C/G/T -> 0..3, every other byte -> its own value (so equality matches
# Biopython char-equality exactly on uppercased sequences). uint8.
_LUT = np.arange(256, dtype=np.uint8)
for b, c in zip(b"ACGT", range(4)):
    _LUT[b] = c

_KERNEL_SRC = f"""
extern "C" {{
#define MATCH ({MATCH})
#define MISMATCH ({MISMATCH})
#define GOPEN ({GAP_OPEN})
#define GEXT ({GAP_EXT})
#define MAXQ ({MAXQ})
#define NEG ({NEG})
#define NEGS (-16000)

__device__ __forceinline__ int sw_score(
    const unsigned char* q, int lq, const unsigned char* r, int lr)
{{
    if (lq > MAXQ) lq = MAXQ;
    short Hp[MAXQ+1];        // DP buffers in int16 to halve local-memory traffic
    short F[MAXQ+1];
    #pragma unroll 1
    for (int j = 0; j <= lq; ++j) {{ Hp[j] = 0; F[j] = NEGS; }}
    int best = 0;
    for (int i = 1; i <= lr; ++i) {{
        unsigned char a = r[i-1];
        int diag = Hp[0];      // H(i-1,0) = 0
        int Hleft = 0;         // H(i,0)   = 0
        int Eleft = NEG;       // E(i,0)
        #pragma unroll 1
        for (int j = 1; j <= lq; ++j) {{
            unsigned char b = q[j-1];
            int Fj = Hp[j] + GOPEN; int Fe = F[j] + GEXT;   Fj = Fj > Fe ? Fj : Fe;
            int Ej = Hleft + GOPEN; int Ee = Eleft + GEXT;  Ej = Ej > Ee ? Ej : Ee;
            int s = (a == b) ? MATCH : MISMATCH;
            int Hij = diag + s;
            if (Hij < 0)  Hij = 0;
            if (Ej > Hij) Hij = Ej;
            if (Fj > Hij) Hij = Fj;
            if (Hij > best) best = Hij;
            diag = Hp[j];
            Hp[j] = (short)Hij; F[j] = (short)Fj;
            Hleft = Hij; Eleft = Ej;
        }}
    }}
    return best;
}}

// cross product: score qlist[0..nq) x all refs[0..nr)  -> out[nq*nr]  (row-major nq x nr)
__global__ void sw_cross(
    const unsigned char* qbuf, const int* qoff,
    const unsigned char* rbuf, const int* roff,
    const int* qlist, int nq, int nr, int* out)
{{
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)nq * nr;
    if (t >= total) return;
    int qi = qlist[(int)(t / nr)];
    int rj = (int)(t % nr);
    int qs = qoff[qi], rs = roff[rj];
    out[t] = sw_score(qbuf + qs, qoff[qi+1] - qs, rbuf + rs, roff[rj+1] - rs);
}}

// arbitrary pairs: score (pair_qi[k], pair_rj[k]) -> out[k]
__global__ void sw_pairs(
    const unsigned char* qbuf, const int* qoff,
    const unsigned char* rbuf, const int* roff,
    const int* pair_qi, const int* pair_rj, int npair, int* out)
{{
    long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= npair) return;
    int qi = pair_qi[k], rj = pair_rj[k];
    int qs = qoff[qi], rs = roff[rj];
    out[k] = sw_score(qbuf + qs, qoff[qi+1] - qs, rbuf + rs, roff[rj+1] - rs);
}}
}}  // extern C
"""

_MOD = cp.RawModule(code=_KERNEL_SRC, options=("--std=c++14",))
_CROSS = _MOD.get_function("sw_cross")
_PAIRS = _MOD.get_function("sw_pairs")


def encode_concat(seqs):
    """list[str] -> (uint8 codes concat, int32 offsets[n+1])."""
    offs = np.zeros(len(seqs) + 1, dtype=np.int32)
    parts = []
    for i, s in enumerate(seqs):
        a = _LUT[np.frombuffer(s.upper().encode("ascii", "ignore"), dtype=np.uint8)]
        parts.append(a)
        offs[i + 1] = offs[i] + a.shape[0]
    return (np.concatenate(parts) if parts else np.zeros(0, np.uint8)), offs


def encode_blob(blob: bytes, off: np.ndarray):
    """Encode the reference blob (bytes) in place -> uint8 codes; offsets to int32."""
    codes = _LUT[np.frombuffer(blob, dtype=np.uint8)]
    return codes, off.astype(np.int32)


class GpuSW:
    def __init__(self, asv_seqs, ref_codes, ref_off):
        qc, qo = encode_concat(asv_seqs)
        self.d_qbuf = cp.asarray(qc); self.d_qoff = cp.asarray(qo)
        self.d_rbuf = cp.asarray(ref_codes); self.d_roff = cp.asarray(ref_off)
        self.nr = ref_off.shape[0] - 1
        self.ref_off = ref_off

    def score_pairs(self, qi, rj):
        qi = cp.asarray(np.asarray(qi, np.int32)); rj = cp.asarray(np.asarray(rj, np.int32))
        n = int(qi.shape[0]); out = cp.empty(n, dtype=cp.int32)
        thr = 128; _PAIRS((( n + thr - 1)//thr,), (thr,),
                          (self.d_qbuf, self.d_qoff, self.d_rbuf, self.d_roff, qi, rj, np.int32(n), out))
        return out

    def score_cross(self, qlist):
        """qlist: array of ASV indices -> int32 scores (len(qlist), nr) on GPU."""
        ql = cp.asarray(np.asarray(qlist, np.int32)); nq = int(ql.shape[0])
        out = cp.empty(nq * self.nr, dtype=cp.int32)
        total = nq * self.nr; thr = 128
        _CROSS(((total + thr - 1)//thr,), (thr,),
               (self.d_qbuf, self.d_qoff, self.d_rbuf, self.d_roff, ql, np.int32(nq), np.int32(self.nr), out))
        return out.reshape(nq, self.nr)


# --------------------------------------------------------------------------- self-test / bench
def main():
    import argparse, csv
    from Bio import Align
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="exhaustive GPU SW over ALL refs for ALL ASVs -> prefilter-free mapping JSON + pipeline comparison")
    ap.add_argument("--asv-batch", type=int, default=24, help="ASVs scored per kernel launch (keeps each launch short)")
    ap.add_argument("--topk", type=int, default=20, help="final hits kept per ASV (full mode)")
    ap.add_argument("--cand", type=int, default=40, help="GPU candidates pulled before deterministic tie-break (full mode)")
    ap.add_argument("--limit", type=int, default=0, help="full mode: restrict to first N ASVs (smoke test)")
    ap.add_argument("--outdir", default="", help="full mode: output dir (default = script dir)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 4), help="full mode: enrichment workers")
    args = ap.parse_args()

    t0 = time.perf_counter()
    print("[load] references + ASVs ...", flush=True)
    md5s, blob, off, ref_len = P.load_references(200)
    ref_codes, ref_off = encode_blob(blob, off)
    asv_ids, asv_seqs = P.load_asvs(P.ASVS_FASTA)
    if args.full and args.limit:
        asv_ids, asv_seqs = asv_ids[:args.limit], asv_seqs[:args.limit]
    id2idx = {a: i for i, a in enumerate(asv_ids)}
    nr = len(md5s)
    maxq = max(len(s) for s in asv_seqs)
    print(f"[load] {nr:,} refs | {len(asv_ids):,} ASVs | max ASV len {maxq} (MAXQ={MAXQ})", flush=True)
    assert maxq <= MAXQ, "increase MAXQ"

    gpu = GpuSW(asv_seqs, ref_codes, ref_off)
    print(f"[gpu] kernel compiled; ref buffer {ref_codes.nbytes/1e6:.0f} MB on device", flush=True)

    if args.full:
        run_full(args, gpu, blob, md5s, off, ref_len, asv_ids, asv_seqs, t0)
        return

    # ---- correctness vs Biopython on random real pairs ----
    aln = Align.PairwiseAligner(); aln.mode = "local"
    aln.match_score = P.MATCH; aln.mismatch_score = P.MISMATCH
    aln.open_gap_score = P.GAP_OPEN; aln.extend_gap_score = P.GAP_EXTEND
    rng = np.random.default_rng(0)
    NQC = 400
    qi = rng.integers(0, len(asv_ids), NQC); rj = rng.integers(0, nr, NQC)
    g = cp.asnumpy(gpu.score_pairs(qi, rj))
    bio = np.array([aln.score(asv_seqs[qi[k]], blob[off[rj[k]]:off[rj[k]+1]].decode()) for k in range(NQC)])
    diff = np.abs(g - bio)
    print(f"[correctness] {NQC} random pairs: exact_match={int((diff==0).sum())}/{NQC} "
          f"max_abs_diff={int(diff.max())} mean_abs_diff={diff.mean():.3f}", flush=True)

    # ---- GPU-exhaustive over all refs for the 20 validation ASVs ----
    val_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_recall_per_asv.csv")
    val_ids = [r["asv"] for r in csv.DictReader(open(val_csv))]
    qlist = [id2idx[a] for a in val_ids]
    # ref residues touched (for GCUPS): each ASV scans all ref bases
    total_ref_bases = int(ref_len.sum())
    cells = sum(len(asv_seqs[i]) for i in qlist) * total_ref_bases
    cp.cuda.runtime.deviceSynchronize(); ts = time.perf_counter()
    scores = gpu.score_cross(qlist)                 # (20, nr)
    cp.cuda.runtime.deviceSynchronize(); gpu_wall = time.perf_counter() - ts
    gcups = cells / gpu_wall / 1e9
    print(f"[exhaustive] {len(qlist)} ASVs x {nr:,} refs in {gpu_wall:.1f}s  "
          f"=> {gcups:.1f} GCUPS  (cells={cells/1e12:.2f}T)", flush=True)

    # top-5 per ASV on GPU, compare md5 sets to CPU-exhaustive validation
    val_json = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "asv_top5_alignment_hits_validation.json")))
    top5_idx = cp.asnumpy(cp.argsort(-scores, axis=1)[:, :5])
    top5_sc = cp.asnumpy(cp.take_along_axis(scores, cp.asarray(top5_idx), axis=1))
    captured = best_match = 0
    rows = []
    for a, asv in enumerate(val_ids):
        gpu_md5 = [md5s[int(j)] for j in top5_idx[a]]
        cpu = val_json[asv]["top5"]
        cpu_md5 = [h["md5"] for h in cpu]
        cpu_best_score = cpu[0]["align_score"]
        inter = len(set(gpu_md5) & set(cpu_md5))
        captured += inter
        bm = (gpu_md5[0] == cpu_md5[0]); best_match += bm
        rows.append((asv[:8], int(top5_sc[a][0]), int(cpu_best_score), inter, bm))
    print(f"\n[validate] GPU-exhaustive vs CPU-exhaustive top-5 over {len(val_ids)} ASVs:")
    print(f"  top-5 md5 overlap: {captured}/{len(val_ids)*5} ({100*captured/(len(val_ids)*5):.1f}%) | "
          f"top-1 md5 match: {best_match}/{len(val_ids)}")
    print("  ASV       gpuTop1  cpuTop1  capt/5  top1match")
    for r in rows:
        print(f"  {r[0]:8} {r[1]:7} {r[2]:8} {r[3]:6}  {r[4]}")

    stats = {
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "n_refs": nr, "n_val_asvs": len(val_ids),
        "kernel": "custom NVRTC affine-local-SW, 1 thread/pair",
        "scoring": {"match": MATCH, "mismatch": MISMATCH, "gap_open": GAP_OPEN, "gap_extend": GAP_EXT, "mode": "local"},
        "correctness_exact_match": f"{int((diff==0).sum())}/{NQC}",
        "correctness_max_abs_diff": int(diff.max()),
        "exhaustive_wall_s": round(gpu_wall, 2),
        "exhaustive_gcups": round(gcups, 1),
        "top5_md5_overlap_pct": round(100*captured/(len(val_ids)*5), 1),
        "top1_md5_match": f"{best_match}/{len(val_ids)}",
        "cpu_exhaustive_wall_s_ref": 511.0,
        "speedup_vs_cpu_exhaustive": round(511.0 / gpu_wall, 1),
        "extrapolated_full_3950_exhaustive_gpu_min": round(gpu_wall * (len(asv_ids)/len(val_ids)) / 60, 1),
        "total_wall_s": round(time.perf_counter() - t0, 1),
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_align_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print("\n[stats]", json.dumps(stats, indent=2))


# --- enrichment-worker globals (set before the Pool; inherited via fork) ---
_E_BLOB = None; _E_OFF = None; _E_ASV = None; _E_TOPIDX = None; _E_TOPSC = None; _E_K = None; _E_ALN = None


def _enr_init():
    global _E_ALN
    _E_ALN = P.make_aligner()


def _enrich_asv(qi):
    """Enrich one ASV's top-K GPU hits with Biopython identity/coords + edlib distance."""
    import edlib
    aseq = _E_ASV[qi]
    cand = sorted(zip(_E_TOPSC[qi].tolist(), _E_TOPIDX[qi].tolist()), key=lambda x: (-x[0], x[1]))[:_E_K]
    out = []
    for sc, idx in cand:
        ref = _E_BLOB[_E_OFF[idx]:_E_OFF[idx + 1]].decode("ascii")
        aln = _E_ALN.align(aseq, ref)[0]
        ident, matches, alen, rs, re_ = P._identity_and_coords(aln)
        ed = edlib.align(aseq, ref, mode="HW", task="distance")["editDistance"]
        out.append((int(idx), float(sc), round(ident, 4), matches, alen, rs, re_, int(ed)))
    return qi, out


def run_full(args, gpu, blob, md5s, off, ref_len, asv_ids, asv_seqs, t0):
    """Exhaustive GPU SW over ALL refs for ALL ASVs -> prefilter-free top-K _gpu mapping + pipeline comparison."""
    import csv
    from multiprocessing import Pool
    global _E_BLOB, _E_OFF, _E_ASV, _E_TOPIDX, _E_TOPSC, _E_K
    here = args.outdir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(here, exist_ok=True)
    nr = gpu.nr; nq = len(asv_ids)
    K = args.topk; CAND = max(args.cand, K + 20); B = args.asv_batch
    print(f"[full] exhaustive {nq:,} ASVs x {nr:,} refs | batch={B} | top-{K}", flush=True)

    # ---- GPU exhaustive in ASV batches (each launch stays short) ----
    top_idx = np.zeros((nq, CAND), np.int32)
    top_sc = np.full((nq, CAND), -1, np.int32)
    pool = cp.get_default_memory_pool()
    ts = time.perf_counter()
    for b0 in range(0, nq, B):
        qb = list(range(b0, min(b0 + B, nq)))
        scores = gpu.score_cross(qb)                       # (len(qb), nr) int32 on GPU
        part = cp.argpartition(-scores, CAND, axis=1)[:, :CAND]
        sc = cp.take_along_axis(scores, part, axis=1)
        top_idx[b0:b0 + len(qb)] = cp.asnumpy(part)
        top_sc[b0:b0 + len(qb)] = cp.asnumpy(sc)
        del scores, part, sc; pool.free_all_blocks()
        done = b0 + len(qb); el = time.perf_counter() - ts
        print(f"[full] {done}/{nq} ASVs  elapsed={el:7.0f}s  eta={el/max(done,1)*(nq-done):7.0f}s", flush=True)
    gpu_wall = time.perf_counter() - ts
    cells = sum(len(s) for s in asv_seqs) * int(ref_len.sum())
    print(f"[full] GPU exhaustive done in {gpu_wall:.0f}s ({cells/gpu_wall/1e9:.1f} GCUPS)", flush=True)

    # ---- parallel enrichment: Biopython identity/coords + edlib distance for all K hits ----
    print(f"[full] enrich {nq * K:,} hits ({K}/ASV) across {args.workers} workers ...", flush=True)
    _E_BLOB, _E_OFF, _E_ASV = blob, off, asv_seqs
    _E_TOPIDX, _E_TOPSC, _E_K = top_idx, top_sc, K
    enr = {}
    te = time.perf_counter()
    with Pool(args.workers, initializer=_enr_init) as pl:
        dn = 0
        for qi, recs in pl.imap_unordered(_enrich_asv, range(nq), chunksize=4):
            enr[qi] = recs; dn += 1
            if dn % 500 == 0:
                print(f"[full] enriched {dn}/{nq}  ({time.perf_counter() - te:.0f}s)", flush=True)
    print(f"[full] enrichment done in {time.perf_counter() - te:.0f}s", flush=True)

    # ---- attach md5 / BV-BRC header metadata / NCBI lineage (parent, cached) ----
    print("[full] md5->header + taxdump ...", flush=True)
    md5_hdr = json.load(open(P.DB_MD5_ID))
    import taxopy
    taxdb = taxopy.TaxDb(nodes_dmp=P.TAXDUMP_NODES, names_dmp=P.TAXDUMP_NAMES)
    lin_cache = {}

    def lineage_for(tid):
        if tid is None:
            return {r: None for r in P.RANKS}
        if tid not in lin_cache:
            try:
                rd = taxopy.Taxon(tid, taxdb).rank_name_dictionary
                lin_cache[tid] = {"Kingdom": rd.get("superkingdom") or rd.get("kingdom") or rd.get("domain"),
                                  "Phylum": rd.get("phylum"), "Class": rd.get("class"), "Order": rd.get("order"),
                                  "Family": rd.get("family"), "Genus": rd.get("genus"), "Species": rd.get("species")}
            except Exception:
                lin_cache[tid] = {r: None for r in P.RANKS}
        return lin_cache[tid]

    midas = P.load_midas(P.TAXONOMY_CSV)
    listkey = f"top{K}"
    mapping = {}
    for qi in range(nq):
        asv = asv_ids[qi]
        hits = []
        for rank, (idx, sc, ident, matches, alen, rs, re_, ed) in enumerate(enr[qi], 1):
            md5 = md5s[idx]
            org, gid, tid, feat = P.parse_header(md5_hdr.get(md5, ""))
            hits.append({"rank": rank, "align_score": sc, "identity": ident,
                         "n_matches": matches, "aligned_len": alen,
                         "edlib_distance": ed, "edlib_identity": round(1 - ed / max(1, len(asv_seqs[qi])), 4),
                         "organism": org, "genome_id": gid, "taxon_id": tid, "feature_id": feat, "md5": md5,
                         "ref_seq_len": int(ref_len[idx]), "ref_aln_start": rs, "ref_aln_end": re_,
                         "lineage": lineage_for(tid)})
        midas_tax, rel_ab = midas.get(asv, ("", 0.0))
        mapping[asv] = {"asv_len": len(asv_seqs[qi]), "midas_taxonomy": midas_tax, "rel_ab": rel_ab,
                        "search": "gpu_exhaustive_local_SW_all_BVBRC", "n_refs_searched": nr,
                        "best_align_score": hits[0]["align_score"] if hits else None,
                        "best_identity": hits[0]["identity"] if hits else None,
                        listkey: hits}
    out_json = os.path.join(here, f"asv_top{K}_alignment_hits_gpu.json")
    with open(out_json, "w") as fh:
        json.dump(mapping, fh)
    print(f"[full] wrote {out_json} ({os.path.getsize(out_json) / 1e6:.0f} MB)", flush=True)

    # ---- full-scale prefilter validation: GPU-exhaustive (truth) vs edlib->biopython pipeline ----
    summary = {}
    DEPTH = 20
    pipe_path = os.path.join(here, "asv_top20_alignment_hits.json")
    if os.path.exists(pipe_path):
        pipe = json.load(open(pipe_path))
        per = []
        tot_overlap = full = top1_in_p20 = top1_match = asvs_with_miss = genuine_miss = pipe_outside_500 = 0
        for asv in asv_ids:
            allg = mapping[asv][listkey]
            g20 = allg[:DEPTH]
            g20_md5 = [h["md5"] for h in g20]
            g500_md5 = {h["md5"] for h in allg}
            p = pipe.get(asv, {}).get("top20", [])
            p_rank = {h["md5"]: h["rank"] for h in p}
            p_min = min((h["align_score"] for h in p), default=float("-inf"))
            ov = len(set(g20_md5) & set(p_rank)); tot_overlap += ov; full += (ov == len(g20_md5))
            t1in = bool(g20_md5) and g20_md5[0] in p_rank; top1_in_p20 += t1in
            t1m = bool(g20_md5) and bool(p) and g20_md5[0] == p[0]["md5"]; top1_match += t1m
            gm = sum(1 for h in g20 if h["md5"] not in p_rank and h["align_score"] > p_min)
            genuine_miss += gm; asvs_with_miss += (gm > 0)
            po = sum(1 for m in p_rank if m not in g500_md5); pipe_outside_500 += po
            per.append({"asv": asv, "gpu_vs_pipe_top20_overlap": ov, "gpu_top1_in_pipeline_top20": t1in,
                        "gpu_top1_is_pipeline_top1": t1m, "prefilter_missed_top20_hits": gm,
                        "pipeline_hits_outside_gpu_top500": po,
                        "gpu_best_score": g20[0]["align_score"] if g20 else None,
                        "pipeline_best_score": p[0]["align_score"] if p else None})
        with open(os.path.join(here, "gpu_vs_pipeline_full_comparison_per_asv.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per[0].keys())); w.writeheader(); w.writerows(per)
        summary = {
            "depth_compared": DEPTH,
            "pct_pipeline_top20_overlap_with_gpu_top20": round(100 * tot_overlap / (nq * DEPTH), 3),
            "pct_asvs_full_20of20_overlap": round(100 * full / nq, 3),
            "pct_asvs_gpu_top1_in_pipeline_top20": round(100 * top1_in_p20 / nq, 3),
            "pct_asvs_gpu_top1_is_pipeline_top1": round(100 * top1_match / nq, 3),
            "n_prefilter_genuine_miss_top20_hits": genuine_miss,
            "n_asvs_with_any_prefilter_miss": asvs_with_miss,
            "n_pipeline_top20_hits_outside_gpu_top500": pipe_outside_500,
        }

    stats = {"gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
             "mode": "full_exhaustive", "n_asvs": nq, "n_refs": nr, "topk": K, "asv_batch": B,
             "gpu_exhaustive_wall_s": round(gpu_wall, 1), "gpu_exhaustive_gcups": round(cells / gpu_wall / 1e9, 1),
             "cpu_exhaustive_extrapolated_h": round(511.0 * (nq / 20) / 3600, 1),
             "total_wall_s": round(time.perf_counter() - t0, 1),
             "comparison_vs_pipeline": summary}
    json.dump(stats, open(os.path.join(here, "gpu_exhaustive_full_stats.json"), "w"), indent=2)
    print("\n[full stats]", json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
