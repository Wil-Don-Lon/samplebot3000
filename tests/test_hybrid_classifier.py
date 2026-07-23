"""Unit tests for the hybrid classifier: taxonomy, hat-articulation mining, and
the cascade/flat inference interface.

The architecture tests fit tiny estimators on synthetic data so they run fast and
offline. Logic-dependent checks skip cleanly when the factory content is absent.

Run:  python -m pytest tests/test_hybrid_classifier.py -q
  or: python tests/test_hybrid_classifier.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drum_data import TAXONOMY, LEAF_TO_FAMILY
import logic_import
import train_hybrid as T


def test_taxonomy_consistency():
    # Every leaf maps back to the family that lists it, and back again.
    for family, leaves in TAXONOMY.items():
        for leaf in leaves:
            assert LEAF_TO_FAMILY[leaf] == family
    assert set(LEAF_TO_FAMILY.values()) == set(TAXONOMY)
    # The families the cascade splits must each have >1 leaf.
    multi = [f for f, ls in TAXONOMY.items() if len(ls) > 1]
    assert set(multi) == {"tom", "cymbal", "hat"}


def test_hat_articulation_keys():
    # GM hi-hat keys: 42/44 closed(+pedal), 46 open.
    a = logic_import.HAT_KEY_ARTICULATION
    assert a[42] == "hats_closed" and a[44] == "hats_closed"
    assert a[46] == "hats_open"


def _synthetic_cache(n_per=50, seed=0):
    rng = np.random.default_rng(seed)
    leaves = list(LEAF_TO_FAMILY)
    y_leaf, groups, hand = [], [], []
    for li, leaf in enumerate(leaves):
        if leaf in ("hats_closed", "hats_open"):
            continue                       # keep the clean 9-leaf set
        for j in range(n_per):
            y_leaf.append(leaf)
            groups.append(f"g{j % 5}")
            hand.append(rng.normal(li, 0.5, size=57))
    return {
        "hand": np.asarray(hand, np.float32),
        "y_leaf": np.asarray(y_leaf),
        "y_family": np.asarray([LEAF_TO_FAMILY[l] for l in y_leaf]),
        "groups": np.asarray(groups),
    }


def test_cascade_and_flat_interface():
    cache = _synthetic_cache()
    fts = ["hand"]
    choices = T.select_choices(cache, ["gb"], fts, n_splits=3)
    leaves = sorted(set(cache["y_leaf"]))
    all_idx = np.arange(len(cache["y_leaf"]))
    feat = {"hand": cache["hand"][:8]}

    for kind, ctor in (("cascade", T.fit_cascade), ("flat", T.fit_flat)):
        model = ctor(cache, all_idx, choices)
        assert sorted(model.classes_) == leaves
        assert model.feature_types_needed() == {"hand"}
        P = model.predict_proba(feat)
        assert P.shape == (8, len(model.classes_))
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
        preds = model.predict(feat)
        assert set(preds) <= set(model.classes_)


def test_fx_rejection_on_flat():
    # With a high tau, low-confidence rows are rejected to "fx".
    cache = _synthetic_cache()
    choices = T.select_choices(cache, ["gb"], ["hand"], n_splits=3)
    model = T.fit_flat(cache, np.arange(len(cache["y_leaf"])), choices)
    model.fx_tau = 2.0                     # impossible confidence -> all rejected
    preds = model.predict({"hand": cache["hand"][:6]})
    assert all(p == "fx" for p in preds)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all hybrid-classifier tests passed")
