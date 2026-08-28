"""
purged_cv.py — cross-validation that does not leak, for overlapping labels.

WHY ORDINARY K-FOLD IS BROKEN HERE
Your label is a 20-day forward return. A sample dated 1 March has a label built
from prices up to 29 March. A sample dated 15 March overlaps it almost
entirely. Standard k-fold will happily put one in train and the other in test,
so the model sees the answer to the test question during training.

The result is not a small optimism. On overlapping financial labels, naive CV
routinely reports IC two to five times the honest value, and the error grows
with the label horizon. Any model validated that way is untrustworthy.

TWO FIXES, BOTH FROM LOPEZ DE PRADO (Advances in Financial ML, ch. 7)

  PURGING   drop from the TRAINING set every sample whose label window overlaps
            the test window. That kills the direct leak.

  EMBARGO   also drop training samples for a short period AFTER the test
            window. Serial correlation in returns and features means a sample
            just after the test block still carries information about it, even
            with no label overlap.

The classes below are drop-in replacements for sklearn's KFold and work with
cross_val_predict, GridSearchCV and friends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PurgedKFold:
    """K-fold over time-ordered samples with purging and an embargo.

    Parameters
    ----------
    n_splits : number of folds
    t1 : Series indexed like X, giving the END time of each sample's label.
         For a 20-day forward return starting at t, t1 = t + 20 trading days.
    embargo_pct : fraction of the total sample embargoed after each test block.
                  0.01 is a common default; larger for slower-moving features.
    """

    def __init__(self, n_splits: int = 5, t1: pd.Series | None = None,
                 embargo_pct: float = 0.01):
        if t1 is None:
            raise ValueError("t1 (label end times) is required — without it "
                             "purging cannot know what overlaps what.")
        self.n_splits = n_splits
        self.t1 = t1
        self.embargo_pct = embargo_pct

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if len(X) != len(self.t1):
            raise ValueError("X and t1 must be the same length")
        idx = np.arange(len(X))
        embargo = int(len(X) * self.embargo_pct)
        # contiguous test blocks, in time order — never shuffled
        bounds = [(b[0], b[-1] + 1)
                  for b in np.array_split(idx, self.n_splits)]
        t1 = self.t1.reset_index(drop=True)

        for start, stop in bounds:
            test = idx[start:stop]
            t0_test = t1.index[start]
            # a training sample is dropped if its label window touches the test
            # window at all, in either direction
            test_start_time = t1.iloc[start] if start < len(t1) else t1.iloc[-1]
            test_first = t1.index[start]
            test_last = t1.index[stop - 1]

            max_t1_in_test = t1.iloc[start:stop].max()
            min_t_in_test = start

            train = []
            for i in idx:
                if start <= i < stop:
                    continue
                # purge: label of i ends after the test block begins, and i
                # starts before the test block's labels finish
                if i < start:
                    if t1.iloc[i] >= t1.index[start]:   # overlaps forward
                        # compare positions, not timestamps, when index is int
                        pass
                    # positional overlap check below handles it
                train.append(i)
            train = np.array(train)

            # positional purge + embargo
            keep = []
            for i in train:
                if i < start:
                    # does sample i's label extend into the test block?
                    # label of i covers positions [i, pos_of_t1_i]
                    if _label_end_pos(t1, i) >= start:
                        continue
                else:
                    # after the test block: apply embargo
                    if i < stop + embargo:
                        continue
                keep.append(i)
            yield np.array(keep), test


def _label_end_pos(t1: pd.Series, i: int) -> int:
    """Position of the bar where sample i's label finishes."""
    end_time = t1.iloc[i]
    # t1 values are timestamps; find how far forward that is in the series
    pos = t1.searchsorted(end_time)
    return int(pos)


def purged_train_test(n: int, t1_pos: np.ndarray, test_start: int,
                      test_stop: int, embargo: int):
    """Simpler positional API used by the pipeline.

    t1_pos[i] = index position at which sample i's label completes.
    Returns (train_idx, test_idx).
    """
    idx = np.arange(n)
    test = idx[test_start:test_stop]
    train = []
    for i in idx:
        if test_start <= i < test_stop:
            continue
        if i < test_start:
            if t1_pos[i] >= test_start:      # label runs into the test block
                continue
        else:
            if i < test_stop + embargo:      # embargo after the block
                continue
        train.append(i)
    return np.array(train, dtype=int), test


def leak_demo(n=3000, horizon=20, seed=0):
    """Demonstrate leakage on data with NO predictable signal.

    Leakage needs a model flexible enough to MEMORISE. A one-feature linear
    model cannot, so it shows nothing. A gradient-boosted tree given a
    locally-identifying feature can memorise "around this point in time the
    label was X" — and because 20-day labels overlap, the neighbouring test
    samples share most of that label. That is the leak.
    """
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    daily = rng.normal(0, 0.01, n + horizon)
    y = np.array([daily[i:i + horizon].sum() for i in range(n)])   # overlapping
    # features: pure noise, plus a locally-identifying one (any slow-moving
    # feature does this in practice — price level, market cap, sector dummy)
    X = np.column_stack([
        rng.normal(0, 1, n),
        np.arange(n) / n + rng.normal(0, 0.001, n),
    ])
    t1_pos = np.minimum(np.arange(n) + horizon, n - 1)

    def run(purge):
        preds = np.full(n, np.nan)
        for k in range(5):
            a, b = k * n // 5, (k + 1) * n // 5
            if purge:
                tr, te = purged_train_test(n, t1_pos, a, b, embargo=horizon)
            else:
                tr = np.r_[np.arange(0, a), np.arange(b, n)]
                te = np.arange(a, b)
            if len(tr) < 100:
                continue
            m = lgb.LGBMRegressor(n_estimators=300, num_leaves=63,
                                  learning_rate=0.1, verbose=-1)
            m.fit(X[tr], y[tr])
            preds[te] = m.predict(X[te])
        ok = ~np.isnan(preds)
        return float(pd.Series(preds[ok]).corr(pd.Series(y[ok]), method="spearman"))

    return {"naive_ic": run(False), "purged_ic": run(True)}


if __name__ == "__main__":
    print("Leakage demonstration — data contains NO real signal, true IC = 0\n")
    print(f"  {'seed':>6}{'naive k-fold':>16}{'purged + embargo':>20}")
    print("  " + "-" * 42)
    naive, purged = [], []
    for s in range(5):
        r = leak_demo(seed=s)
        naive.append(r["naive_ic"]); purged.append(r["purged_ic"])
        print(f"  {s:>6}{r['naive_ic']:>15.4f}{r['purged_ic']:>19.4f}")
    print("  " + "-" * 42)
    print(f"  {'mean':>6}{np.mean(naive):>15.4f}{np.mean(purged):>19.4f}")
    print("\n  True IC is zero. Whatever naive CV reports above zero is leakage.")
