"""Hybrid drum-classifier architectures (built + selected by train_hybrid).

Two containers, one interface, so either can drop into the serving path
(`pipeline.classify_segments` / `cluster_consensus`) and be A/B'd fairly:

    predict(feat)        -> (n,) leaf labels (may be "fx" if rejection is on)
    predict_proba(feat)  -> (n, n_leaves) aligned to .classes_ (rows sum to 1)
    .classes_            -> leaf labels
    feature_types_needed()-> which representations `feat` must provide

`feat` is a dict mapping feature_type ("hand" | "fused") -> matrix, so every
internal node can use the representation that won its bake-off. FX is open-set
rejection (max confidence < tau -> "fx"), never a trained class.

- CascadeClassifier: family clf -> per-family fine clf; P(leaf)=P(fam)·P(leaf|fam).
- FlatSpecialistClassifier: calibrated one-vs-rest per leaf; argmax P(leaf).

Fitting/selection lives in train_hybrid; these classes just hold the chosen,
already-fitted estimators and implement inference + are joblib-picklable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Node:
    """A fitted estimator plus the feature representation it consumes."""
    estimator: object            # fitted sklearn pipeline with predict_proba
    feature_type: str            # "hand" | "fused"
    positive_label: Optional[str] = None   # set for binary one-vs-rest specialists

    def proba(self, feat: dict) -> tuple[np.ndarray, np.ndarray]:
        X = feat[self.feature_type]
        return self.estimator.predict_proba(X), np.asarray(self.estimator.classes_)

    def positive_proba(self, feat: dict) -> np.ndarray:
        P, classes = self.proba(feat)
        j = int(np.where(classes == self.positive_label)[0][0])
        return P[:, j]


class CascadeClassifier:
    architecture = "cascade"

    def __init__(self, family_node: Node, fine_nodes: dict[str, Node],
                 classes: list[str], fx_node: Optional[Node] = None,
                 fx_tau: Optional[float] = None, meta: Optional[dict] = None):
        self.family_node = family_node
        self.fine_nodes = fine_nodes            # family -> Node (only multi-leaf families)
        self.classes_ = np.asarray(classes)
        self.fx_node = fx_node                  # calibrated family clf for rejection
        self.fx_tau = fx_tau
        self.meta = meta or {}

    def feature_types_needed(self) -> set[str]:
        fts = {self.family_node.feature_type}
        fts.update(n.feature_type for n in self.fine_nodes.values())
        if self.fx_node is not None:
            fts.add(self.fx_node.feature_type)
        return fts

    def predict_proba(self, feat: dict) -> np.ndarray:
        n = len(next(iter(feat.values())))
        fam_P, fam_classes = self.family_node.proba(feat)
        col = {c: i for i, c in enumerate(self.classes_)}
        out = np.zeros((n, len(self.classes_)), dtype=float)
        for fi, fam in enumerate(fam_classes):
            p_fam = fam_P[:, fi]
            if fam in self.fine_nodes:
                leaf_P, leaf_classes = self.fine_nodes[fam].proba(feat)
                for li, leaf in enumerate(leaf_classes):
                    if leaf in col:
                        out[:, col[leaf]] += p_fam * leaf_P[:, li]
            elif fam in col:                    # single-leaf family: leaf == family
                out[:, col[fam]] += p_fam
        row = out.sum(axis=1, keepdims=True)
        return np.divide(out, row, out=np.zeros_like(out), where=row > 0)

    def predict(self, feat: dict) -> np.ndarray:
        proba = self.predict_proba(feat)
        pred = self.classes_[proba.argmax(axis=1)]
        if self.fx_node is not None and self.fx_tau is not None:
            fam_P, _ = self.fx_node.proba(feat)
            pred = np.where(fam_P.max(axis=1) < self.fx_tau, "fx", pred)
        return pred.astype(object)


class FlatSpecialistClassifier:
    architecture = "flat"

    def __init__(self, specialists: dict[str, Node], classes: list[str],
                 fx_tau: Optional[float] = None, meta: Optional[dict] = None):
        self.specialists = specialists          # leaf -> binary one-vs-rest Node
        self.classes_ = np.asarray(classes)
        self.fx_tau = fx_tau
        self.meta = meta or {}

    def feature_types_needed(self) -> set[str]:
        return {n.feature_type for n in self.specialists.values()}

    def _leaf_scores(self, feat: dict) -> np.ndarray:
        n = len(next(iter(feat.values())))
        scores = np.zeros((n, len(self.classes_)), dtype=float)
        for i, leaf in enumerate(self.classes_):
            scores[:, i] = self.specialists[leaf].positive_proba(feat)
        return scores

    def predict_proba(self, feat: dict) -> np.ndarray:
        scores = self._leaf_scores(feat)
        row = scores.sum(axis=1, keepdims=True)
        return np.divide(scores, row, out=np.zeros_like(scores), where=row > 0)

    def predict(self, feat: dict) -> np.ndarray:
        scores = self._leaf_scores(feat)
        pred = self.classes_[scores.argmax(axis=1)]
        if self.fx_tau is not None:
            pred = np.where(scores.max(axis=1) < self.fx_tau, "fx", pred)
        return pred.astype(object)
