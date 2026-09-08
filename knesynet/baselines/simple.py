from __future__ import annotations
import numpy as np

class FrequencyBaseline:
    def fit(self, y):
        self.freq = np.asarray(y).mean(0)
        return self
    def predict_scores(self, n):
        return np.tile(self.freq, (n, 1))

class GroupFrequencyBaseline:
    def fit(self, groups, y):
        self.global_freq = np.asarray(y).mean(0)
        self.by_group = {}
        for g in sorted(set(groups)):
            idx = np.asarray(groups) == g
            self.by_group[g] = np.asarray(y)[idx].mean(0)
        return self
    def predict_scores(self, groups):
        return np.stack([self.by_group.get(g, self.global_freq) for g in groups])

class FixedRuleBaseline:
    def predict_scores(self, guide=None, target=None, contra=None):
        out = 0.0
        if guide is not None: out = out + guide
        if target is not None: out = out + target
        if contra is not None: out = out + contra
        return 1.0 / (1.0 + np.exp(-np.asarray(out)))
