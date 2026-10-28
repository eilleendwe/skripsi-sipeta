"""
Random Forest.
Fungsi `fit` bisa diberi `progress` untuk progress bar.
"""

import time
from collections import Counter

import numpy as np


# ============================================================
# NODE
# ============================================================
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


# ============================================================
# DECISION TREE
# ============================================================
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None,
                 max_thresholds=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        # max_thresholds: opsional, None = cek semua nilai unik
        self.max_thresholds = max_thresholds
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            if self.max_thresholds is not None and len(thresholds) > self.max_thresholds:
                percentiles = np.linspace(0, 100, self.max_thresholds)
                thresholds = np.unique(np.percentile(X_column, percentiles))

            for thr in thresholds:
                gain = self._gini_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    # ========================================================
    # GINI GAIN
    # ========================================================
    def _gini_gain(self, y, X_column, threshold):
        parent_gini = self._gini(y)
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return -1  # split tidak valid (salah satu sisi kosong) -> -1 supaya tidak pernah terpilih

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)

        gini_left = self._gini(y[left_idxs])
        gini_right = self._gini(y[right_idxs])

        child_gini = (n_l / n) * gini_left + (n_r / n) * gini_right
        return parent_gini - child_gini

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    # ========================================================
    # Gini IMPURITY
    # ========================================================
    def _gini(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return 1 - np.sum(ps ** 2)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        # x[feature] <= threshold -> kiri. Tiap simpul langsung memproses semua baris yang sampai ke situ, bukan baris per baris.
        out = np.zeros(len(X), dtype=np.int64)
        stack = [(self.root, np.arange(len(X)))]
        while stack:
            node, idx = stack.pop()
            if len(idx) == 0:
                continue
            if node.is_leaf_node():
                out[idx] = node.value
                continue
            go_left = X[idx, node.feature] <= node.threshold
            stack.append((node.left, idx[go_left]))
            stack.append((node.right, idx[~go_left]))
        return out


# ============================================================
# RANDOM FOREST
# ============================================================
class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2,
                 n_features=None, max_thresholds=None, bootstrap=True, verbose=False):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.max_thresholds = max_thresholds
        self.bootstrap = bootstrap
        self.verbose = verbose
        self.trees = []

    def fit(self, X, y, progress=None):
        self.trees = []

        for i in range(self.n_trees):
            t0 = time.time()

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features,
                max_thresholds=self.max_thresholds,
            )

            # Bootstrap sampling
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_samples(X, y)
            else:
                X_sample, y_sample = X, y

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

            if self.verbose:
                print(f"    [pohon {i+1}/{self.n_trees}] selesai ({time.time()-t0:.1f} detik)")
            if progress:
                progress(i + 1, self.n_trees)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        # Voting mayoritas label 0/1. Kalau seri, ikut prediksi pohon pertama.
        predictions = np.array([tree.predict(X) for tree in self.trees])  # (n_trees, n_baris)
        n_satu = predictions.sum(axis=0)
        n_nol = len(self.trees) - n_satu
        hasil = (n_satu > n_nol).astype(np.int64)
        seri = n_satu == n_nol
        hasil[seri] = predictions[0, seri]
        return hasil


# ============================================================
# EVALUASI
# ============================================================
def evaluasi(y_true, y_pred):
    """Confusion matrix [[TN, FP], [FN, TP]] (baris = aktual) dan metriknya"""
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "confusion": [[tn, fp], [fn, tp]],
        "metrik": {
            "accuracy": (tp + tn) / total if total else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        },
    }
