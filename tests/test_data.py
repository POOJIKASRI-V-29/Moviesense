import numpy as np
import pandas as pd
import pytest

from moviesense.data.augment import eval_condition_texts, training_texts
from moviesense.data.splitting import (assert_no_group_leakage, build_groups,
                                       group_stratified_split)
from moviesense.data.text import (InputValidationError, normalise, normalise_title,
                                  split_sentences, text_hash, validate_user_text)
from moviesense.models.metrics import apply_thresholds, bootstrap_macro_f1, tune_thresholds


def test_normalise_unescapes_html_and_collapses_whitespace():
    assert normalise("Ren&eacute;  Belloq\t\x07 escapes") == "René Belloq escapes"


@pytest.mark.parametrize("bad", ["", "   ", "12345 !!!", None, "a" * 2001])
def test_validate_rejects(bad):
    with pytest.raises(InputValidationError):
        validate_user_text(bad)


def test_validate_accepts_and_normalises():
    assert validate_user_text("  A detective\n investigates.  ") == "A detective investigates."


def test_split_sentences_handles_abbreviations_and_initials():
    text = "Dr. Jones meets J. Smith in Cairo. They escape! Then Mr. Brody arrives."
    assert split_sentences(text) == ["Dr. Jones meets J. Smith in Cairo.", "They escape!",
                                     "Then Mr. Brody arrives."]


def test_text_hash_ignores_case_and_punctuation():
    assert text_hash("The Hero, returns!") == text_hash("the hero returns")


def test_normalise_title_groups_variants():
    assert normalise_title("The Thing (1982)") == normalise_title("Thing")


def test_build_groups_merges_titles_and_duplicates():
    g = build_groups(["Hamlet", "Hamlet", "Other", "Dup A", "Dup B"], [(3, 4, 0.9)])
    assert g[0] == g[1] and g[3] == g[4] and len({g[0], g[2], g[3]}) == 3


def test_group_split_keeps_groups_together_and_covers_labels():
    rng = np.random.default_rng(0)
    n = 600
    Y = (rng.random((n, 4)) < [0.5, 0.3, 0.2, 0.1]).astype(int)
    Y[Y.sum(1) == 0, 0] = 1
    groups = np.arange(n) // 2  # pairs of rows share a group
    split = group_stratified_split(Y, groups, seed=1)
    df = pd.DataFrame({"group": groups, "split": split})
    assert_no_group_leakage(df)
    frac = df.split.value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.03
    for s in ("train", "val", "test"):
        assert (Y[split == s].sum(0) > 0).all()


def test_windows_inherit_movie_and_are_deterministic():
    df = pd.DataFrame({"wiki_id": [1], "plot": ["One sentence is here now. Two sentences are here. "
                                                 "Three of them exist now. Four is the last one."]})
    a = training_texts(df, n_windows=2, seed=7)
    b = training_texts(df, n_windows=2, seed=7)
    assert a.equals(b) and (a.wiki_id == 1).all() and list(a.kind) == ["full", "window", "window"]
    assert eval_condition_texts(df, "first") == ["One sentence is here now."]


def test_threshold_tuning_and_fallback():
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 0]])
    p = np.array([[0.9, 0.1], [0.6, 0.2], [0.2, 0.3], [0.1, 0.05]])
    taus = tune_thresholds(y, p)
    pred = apply_thresholds(p, taus)
    assert pred[:3].tolist() == [[1, 0], [1, 0], [0, 1]]
    assert pred.sum(1).min() >= 1  # fallback: at least one genre per row


def test_bootstrap_difference_detects_better_model():
    rng = np.random.default_rng(0)
    y = (rng.random((400, 3)) < 0.4).astype(int)
    good = y.copy()
    bad = (rng.random(y.shape) < 0.4).astype(int)
    out = bootstrap_macro_f1(y, [bad, good], n_boot=200)
    assert out["differences_vs_first"][0]["excludes_zero"]
    assert out["models"][1]["mean"] > out["models"][0]["mean"]
