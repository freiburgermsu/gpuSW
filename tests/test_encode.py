"""The input funnel: every supported form encodes to the same discrete buffers."""
import gzip

import numpy as np
import pytest

from gpusw import Scheme
from gpusw.encode import build_lut, funnel, read_fasta
from gpusw.errors import EncodeError

DNA = Scheme(mode="local")


def _codes(enc, i):
    return enc.codes[enc.offsets[i]:enc.offsets[i + 1]].tolist()


def test_list_dict_pairs_agree():
    seqs = ["ACGT", "TTGGCC", "A"]
    e_list = funnel(seqs, DNA)
    e_dict = funnel({"q0": "ACGT", "q1": "TTGGCC", "q2": "A"}, DNA)
    e_pairs = funnel([("q0", "ACGT"), ("q1", "TTGGCC"), ("q2", "A")], DNA)
    assert e_list.ids == ["q0", "q1", "q2"] == e_dict.ids == e_pairs.ids
    for i in range(3):
        assert _codes(e_list, i) == _codes(e_dict, i) == _codes(e_pairs, i)
    assert e_list.lengths.tolist() == [4, 6, 1]
    assert e_list.offsets.tolist() == [0, 4, 10, 11]


def test_lut_codes_and_sentinel():
    lut, sentinel = build_lut(DNA)
    assert sentinel == 4  # len("ACGT")
    assert [lut[ord(c)] for c in "ACGT"] == [0, 1, 2, 3]
    assert lut[ord("N")] == 4 and lut[ord("X")] == 4


def test_case_insensitive():
    e = funnel(["acgt", "AcGt"], DNA)
    assert _codes(e, 0) == _codes(e, 1) == [0, 1, 2, 3]


def test_case_sensitive_scheme_lowercases_to_sentinel():
    sch = Scheme(mode="local", case_insensitive=False)
    e = funnel(["acgt"], sch)
    assert _codes(e, 0) == [4, 4, 4, 4]  # lowercase not in alphabet -> sentinel


def test_unknown_error_policy_raises():
    sch = Scheme(mode="local", unknown="error")
    with pytest.raises(EncodeError):
        funnel(["ACGTN"], sch)


def test_fasta_file(tmp_path):
    p = tmp_path / "q.fasta"
    p.write_text(">seqA desc here\nACGT\nACGT\n>seqB\nTTTT\n")
    e = funnel(str(p), DNA)
    assert e.ids == ["seqA", "seqB"]
    assert _codes(e, 0) == [0, 1, 2, 3, 0, 1, 2, 3]
    assert e.lengths.tolist() == [8, 4]


def test_fasta_gz(tmp_path):
    p = tmp_path / "q.fasta.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(">s1\nACGT\n")
    assert read_fasta(str(p)) == [("s1", "ACGT")]


def test_fasta_text_vs_raw_sequence():
    e_text = funnel(">x\nACGT", DNA)
    assert e_text.ids == ["x"] and _codes(e_text, 0) == [0, 1, 2, 3]
    e_raw = funnel("ACGT", DNA)
    assert e_raw.ids == ["q0"] and _codes(e_raw, 0) == [0, 1, 2, 3]


def test_preencoded_passthrough():
    codes = np.array([0, 1, 2, 3, 3, 2], dtype=np.uint8)
    offsets = np.array([0, 4, 6], dtype=np.int32)
    e = funnel((codes, offsets), DNA)
    assert e.n == 2 and _codes(e, 1) == [3, 2]


def test_seqrecord_like():
    class Rec:
        def __init__(self, i, s):
            self.id, self.seq = i, s

    e = funnel([Rec("a", "ACGT"), Rec("b", "TT")], DNA)
    assert e.ids == ["a", "b"] and e.lengths.tolist() == [4, 2]


def test_empty_sequences_ok():
    e = funnel(["", "ACGT", ""], DNA)
    assert e.lengths.tolist() == [0, 4, 0]
    assert e.offsets.tolist() == [0, 0, 4, 4]
