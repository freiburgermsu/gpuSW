"""Matrix construction, NCBI parsing, sentinel packing, and bundled matrices."""
import numpy as np
import pytest

from gpusw import Matrix, available_matrices
from gpusw.errors import SchemeError


def test_from_match_mismatch():
    m = Matrix.from_match_mismatch(2, -3, "ACGT")
    assert m.array.shape == (4, 4)
    assert np.array_equal(np.diag(m.array), [2, 2, 2, 2])
    off = m.array[~np.eye(4, dtype=bool)]
    assert (off == -3).all()


def test_named_blosum62_symmetry_and_diag():
    m = Matrix.named("BLOSUM62")
    assert m.array.shape == (len(m.alphabet),) * 2
    assert np.array_equal(m.array, m.array.T)  # symmetric
    w = m.alphabet.index("W")
    assert m.array[w, w] == 11  # BLOSUM62 W:W


def test_named_aliases():
    assert Matrix.named("blosum-62").alphabet == Matrix.named("BLOSUM62").alphabet
    assert Matrix.named("ednafull").array.shape[0] == len(Matrix.named("EDNAFULL").alphabet)


def test_named_unknown_raises():
    with pytest.raises(SchemeError):
        Matrix.named("NOPE")


def test_from_ncbi_roundtrip():
    text = "\n".join([
        "# tiny",
        "  A  C  G  T",
        "A  1 -1 -1 -1",
        "C -1  1 -1 -1",
        "G -1 -1  1 -1",
        "T -1 -1 -1  1",
    ])
    m = Matrix.from_ncbi(text)
    assert m.alphabet == "ACGT"
    assert np.array_equal(np.diag(m.array), [1, 1, 1, 1])


def test_packed_sentinel_mismatch_and_zero():
    m = Matrix.from_match_mismatch(2, -3, "ACGT")
    _, packed = m.packed("mismatch")
    assert packed.shape == (5, 5)
    assert (packed[4, :] == -3).all() and (packed[:, 4] == -3).all()
    _, pz = m.packed("zero")
    assert (pz[4, :] == 0).all() and (pz[:, 4] == 0).all()


def test_hash_changes_with_content():
    a = Matrix.from_match_mismatch(2, -3, "ACGT")
    b = Matrix.from_match_mismatch(2, -4, "ACGT")
    assert a.hash() != b.hash()
    assert a.hash() == Matrix.from_match_mismatch(2, -3, "ACGT").hash()


def test_bad_shape_raises():
    with pytest.raises(SchemeError):
        Matrix("ACGT", np.zeros((3, 3), dtype=int))


def test_bundled_set_present():
    names = set(available_matrices())
    assert {"BLOSUM62", "BLOSUM80", "PAM250", "EDNAFULL"} <= names
