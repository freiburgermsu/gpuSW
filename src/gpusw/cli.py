"""``gpusw`` command-line interface — score FASTA files on the GPU.

    gpusw cross  --queries q.fasta --refs r.fasta --scheme dna --out scores.csv
    gpusw topk   --queries q.fasta --refs r.fasta --scheme blosum62 -k 5 --out hits.csv
    gpusw pairs  --queries q.fasta --refs r.fasta --scheme dna --out pairs.csv
    gpusw info
"""
from __future__ import annotations

import argparse
import csv
import sys

from . import (
    __version__,
    _build_scheme,
    align_pairs,
    align_score,
    available_matrices,
    gpu_available,
)
from .aligner import Aligner
from .encode import read_fasta


def _scoring_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--queries", required=True, help="FASTA file of query sequences")
    p.add_argument("--refs", required=True, help="FASTA file of reference sequences")
    p.add_argument("--scheme", default="dna", help="preset name (dna, blosum62, ...)")
    p.add_argument("--mode", choices=["local", "global", "semiglobal"], default=None)
    p.add_argument("--matrix", default=None, help="named substitution matrix override")
    p.add_argument("--match", type=int, default=None)
    p.add_argument("--mismatch", type=int, default=None)
    p.add_argument("--gap-open", type=int, default=None, dest="gap_open")
    p.add_argument("--gap-extend", type=int, default=None, dest="gap_extend")
    p.add_argument("--dtype", choices=["auto", "int16", "int32"], default=None)
    p.add_argument("--threads", type=int, default=128)
    p.add_argument("--out", "-o", default="-", help="output CSV path ('-' = stdout)")


def _scheme_kw(args) -> dict:
    return dict(
        scheme=args.scheme, mode=args.mode, matrix=args.matrix, match=args.match,
        mismatch=args.mismatch, gap_open=args.gap_open, gap_extend=args.gap_extend,
        dtype=args.dtype,
    )


def _writer(path):
    fh = sys.stdout if path == "-" else open(path, "w", newline="")
    return fh, csv.writer(fh)


def _cmd_cross(args) -> int:
    qs = read_fasta(args.queries)
    rs = read_fasta(args.refs)
    res = align_score(qs, rs, threads=args.threads, **_scheme_kw(args))
    fh, w = _writer(args.out)
    w.writerow(["query", "reference", "score"])
    for qi, qid in enumerate(res.query_ids):
        for rj, rid in enumerate(res.reference_ids):
            w.writerow([qid, rid, int(res.scores[qi, rj])])
    if fh is not sys.stdout:
        fh.close()
    return 0


def _cmd_topk(args) -> int:
    qs = read_fasta(args.queries)
    rs = read_fasta(args.refs)
    al = Aligner(_build_scheme(**_scheme_kw(args)), threads=args.threads)
    al.index(rs).set_queries(qs)
    res = al.top_k(k=args.k)
    fh, w = _writer(args.out)
    w.writerow(["query", "rank", "reference", "score"])
    for qid, hits in res.top_k(args.k):
        for rank, (rid, sc) in enumerate(hits, 1):
            w.writerow([qid, rank, rid, sc])
    if fh is not sys.stdout:
        fh.close()
    sys.stderr.write(f"[gpusw] {al.gcups:.1f} GCUPS\n")
    return 0


def _cmd_pairs(args) -> int:
    qs = read_fasta(args.queries)
    rs = read_fasta(args.refs)
    res = align_pairs(qs, rs, threads=args.threads, **_scheme_kw(args))
    fh, w = _writer(args.out)
    w.writerow(["query", "reference", "score"])
    for k, sc in enumerate(res.scores):
        w.writerow([qs[k][0], rs[k][0], int(sc)])
    if fh is not sys.stdout:
        fh.close()
    return 0


def _cmd_info(_args) -> int:
    print(f"gpusw {__version__}")
    print(f"GPU available: {gpu_available()}")
    if gpu_available():
        import cupy as cp

        print(f"device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        print(f"compute capability: {cp.cuda.Device().compute_capability}")
    print("bundled matrices:", ", ".join(sorted(available_matrices())))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gpusw", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"gpusw {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("cross", help="all-pairs score matrix")
    _scoring_args(pc)
    pc.set_defaults(func=_cmd_cross)

    pt = sub.add_parser("topk", help="per-query top-k references")
    _scoring_args(pt)
    pt.add_argument("-k", type=int, default=5)
    pt.set_defaults(func=_cmd_topk)

    pp = sub.add_parser("pairs", help="score queries vs refs 1:1")
    _scoring_args(pp)
    pp.set_defaults(func=_cmd_pairs)

    pi = sub.add_parser("info", help="environment / GPU info")
    pi.set_defaults(func=_cmd_info)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
