"""CLI smoke tests: argument parsing + info are GPU-free; scoring subcommands need a GPU."""
import csv

import pytest

from gpusw import __version__
from gpusw.cli import main


def _fasta(tmp_path, name, recs):
    p = tmp_path / name
    p.write_text("".join(f">{i}\n{s}\n" for i, s in recs))
    return str(p)


def _read_csv(path):
    with open(path) as fh:
        return list(csv.reader(fh))


# ---- GPU-free: argparse + info -------------------------------------------------
def test_info_runs(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert __version__ in out
    assert "bundled matrices:" in out
    assert "BLOSUM62" in out


def test_missing_required_arg_exits_2():
    with pytest.raises(SystemExit) as e:
        main(["cross", "--refs", "r.fasta"])  # missing --queries
    assert e.value.code == 2


def test_no_subcommand_exits():
    with pytest.raises(SystemExit):
        main([])


# ---- GPU: the scoring subcommands ---------------------------------------------
@pytest.mark.gpu
def test_cli_cross(tmp_path):
    q = _fasta(tmp_path, "q.fasta", [("qa", "ACGTACGT"), ("qb", "TTTT")])
    r = _fasta(tmp_path, "r.fasta", [("ra", "ACGTACGT"), ("rb", "ACGT")])
    out = str(tmp_path / "cross.csv")
    assert main(["cross", "--queries", q, "--refs", r, "--scheme", "dna", "-o", out]) == 0
    rows = _read_csv(out)
    assert rows[0] == ["query", "reference", "score"]
    assert len(rows) == 1 + 2 * 2  # header + 2x2
    scores = {(a, b): int(s) for a, b, s in rows[1:]}
    assert scores[("qa", "ra")] == 16  # identical 8-mer, match +2


@pytest.mark.gpu
def test_cli_topk(tmp_path):
    q = _fasta(tmp_path, "q.fasta", [("qa", "ACGTACGT")])
    r = _fasta(tmp_path, "r.fasta", [("ra", "ACGTACGT"), ("rb", "TTTTTTTT"), ("rc", "ACGT")])
    out = str(tmp_path / "topk.csv")
    assert main(["topk", "--queries", q, "--refs", r, "--scheme", "dna", "-k", "2", "-o", out]) == 0
    rows = _read_csv(out)
    assert rows[0] == ["query", "rank", "reference", "score"]
    assert len(rows) == 1 + 2  # header + k=2 rows
    assert rows[1][2] == "ra"   # best hit is the identical sequence


@pytest.mark.gpu
def test_cli_pairs(tmp_path):
    q = _fasta(tmp_path, "q.fasta", [("qa", "ACGT"), ("qb", "TTTT")])
    r = _fasta(tmp_path, "r.fasta", [("ra", "ACGT"), ("rb", "TTTA")])
    out = str(tmp_path / "pairs.csv")
    assert main(["pairs", "--queries", q, "--refs", r, "--scheme", "dna", "-o", out]) == 0
    rows = _read_csv(out)
    assert rows[0] == ["query", "reference", "score"]
    assert [int(r[2]) for r in rows[1:]] == [8, 6]


def test_cli_missing_file_errors(tmp_path):
    r = _fasta(tmp_path, "r.fasta", [("ra", "ACGT")])
    with pytest.raises(FileNotFoundError):
        main(["cross", "--queries", str(tmp_path / "nope.fasta"), "--refs", r, "--scheme", "dna"])
