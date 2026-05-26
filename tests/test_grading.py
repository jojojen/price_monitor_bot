from __future__ import annotations

from tcg_tracker.grading import looks_like_graded


def test_looks_like_graded_psa_variants() -> None:
    # All variants of PSA grade — "PSA10", "PSA 10", with rest of title
    assert looks_like_graded("UNION ARENA 綾波レイ PSA10")
    assert looks_like_graded("PSA 10 ピカチュウ 限定品")
    assert looks_like_graded("リザードンex PSA9.5")


def test_looks_like_graded_bgs_cgc() -> None:
    assert looks_like_graded("Charizard BGS 9.5")
    assert looks_like_graded("BGS10 sealed")
    assert looks_like_graded("Pikachu CGC 10")
    assert looks_like_graded("CGC9 raw")


def test_looks_like_graded_japanese_markers() -> None:
    assert looks_like_graded("ピカチュウ 鑑定済")
    assert looks_like_graded("リザードンex 鑑定品")
    assert looks_like_graded("PSA10 鑑定書付き")
    assert looks_like_graded("鑑定カード")


def test_looks_like_graded_negative_cases() -> None:
    # Raw card titles — must NOT be detected as graded
    assert not looks_like_graded("UNION ARENA 綾波レイ")
    assert not looks_like_graded("ピカチュウex SAR")
    assert not looks_like_graded("リザードンex SV2a")
    # Empty / None safe
    assert not looks_like_graded("")
    assert not looks_like_graded(None)
    # "PSA" in unrelated context (no grade number)
    assert not looks_like_graded("PSAサポーター")  # PSA without grade digits


def test_looks_like_graded_handles_mixed_text() -> None:
    """Real-world Mercari title shapes mixing Japanese + grade markers."""
    assert looks_like_graded("【PSA10】UNION ARENA AP 綾波レイ")
    assert looks_like_graded("ポケモンカード リザードンex SAR 鑑定済 PSA10")
