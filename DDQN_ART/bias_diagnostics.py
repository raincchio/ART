"""
Empirical verification of the ART mean-Bellman-error reduction theorem.

Theoretical prediction (derived from paper framework):
  At each target sync, the mean Bellman error of bias_net should satisfy:
      mean_be(bias_net) ≈ gamma * mean_be(Q_online)

This file adds a diagnostic wrapper around DDQNAgent that tracks:
  1. mean_be(Q_online) and mean_be(bias_net) at every sync point
  2. ratio mean_be(bias_net) / mean_be(Q_online)  -- predicted to be ≈ gamma
  3. Whether the reduction cumulates across cycles
"""

import os
import csv
import copy
import numpy as np
import torch
import torch.nn.functional as F


class ARTDiagnosticWrapper:
    """
    Wraps DDQNAgent and records per-sync Bellman-error diagnostics.

    The key quantities tracked at every hard-sync step:
      - be_Q1: mean |Q_online(s,a) - y|  (standard DDQN target)
      - be_Q2: mean |bias_net(s,a) - y|  (ART target, y - bias)
      - be_ratio: be_Q2 / be_Q1           (predicted ≈ gamma = 0.99)
      - bias_ema: the running EMA scalar

    If be_ratio < 1 consistently, the theorem prediction holds empirically.
    If be_ratio ≈ gamma, the exact quantitative prediction holds.
    """

    def __init__(self, agent, gamma: float, log_path: str):
        self.agent = agent
        self.gamma = gamma
        self.log_path = log_path
        self.records = []
        self._sync_count = 0

        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
        self._f = open(log_path, "w", newline="")
        writer = csv.writer(self._f)
        writer.writerow([
            "sync_step", "training_steps",
            "mean_td1", "std_td1",         # signed: E[Q1-y]
            "mean_td2", "std_td2",         # signed: E[Q2-y], predicted ≈ gamma*mean_td1
            "signed_ratio",                # mean_td2 / mean_td1, predicted ≈ gamma
            "abs_be_q1", "abs_be_q2",      # absolute Bellman errors for reference
            "bias_ema",
            "gamma",
            "ratio_vs_gamma",              # signed_ratio / gamma  (1.0 = perfect)
        ])
        self._f.flush()
        self._writer = writer

    def measure_bellman_errors(self, replay, batch_size=512):
        """
        Draw a large batch and compute Bellman errors for Q_online and bias_net.

        Theory predicts (signed mean):
            mean(Q2 - y) ≈ gamma * mean(Q1 - y)
        i.e. bias_ema of Q2 ≈ gamma * bias_ema of Q1.

        We track both signed mean (the theoretical quantity) and absolute mean.
        """
        agent = self.agent
        if not agent.art:
            return None

        batch = replay.sample(min(batch_size, len(replay)))
        state, action, next_state, reward, not_done = batch

        with torch.no_grad():
            # Self-bootstrap target: use Q_online itself (not target_net) to compute
            # y_self ≈ T*Q_online — the true Bellman operator applied to Q_online.
            # This is what the theory is about: epsilon_Q = Q - T*Q.
            next_q_self = agent.online_net(next_state).max(dim=1, keepdim=True)[0]
            y_self = reward + not_done * agent.gamma * next_q_self  # T*Q_online

            q1 = agent.online_net(state).gather(1, action.unsqueeze(1))
            td1 = (q1 - y_self).squeeze(1)   # signed epsilon_Q1 = Q1 - T*Q1

            q2 = agent.bias_net(state).gather(1, action.unsqueeze(1))
            td2 = (q2 - y_self).squeeze(1)   # bias_net error vs same operator

        return {
            "mean_td1":     float(td1.mean()),       # ≈ bias_ema
            "std_td1":      float(td1.std()),
            "mean_td2":     float(td2.mean()),       # predicted ≈ gamma * mean_td1
            "std_td2":      float(td2.std()),
            "abs_be_q1":    float(td1.abs().mean()), # absolute Bellman error Q1
            "abs_be_q2":    float(td2.abs().mean()), # absolute Bellman error Q2
        }

    def record_sync(self, training_step: int, replay, batch_size=512):
        """Call this right before (or after) each hard target update."""
        self._sync_count += 1
        m = self.measure_bellman_errors(replay, batch_size)
        if m is None:
            return

        bias_ema = float(self.agent.bias)
        # Signed ratio: mean_td2 / mean_td1 — theoretical prediction = gamma
        signed_ratio = m["mean_td2"] / (m["mean_td1"] + 1e-8) if abs(m["mean_td1"]) > 1e-6 else float("nan")
        ratio_vs_gamma = signed_ratio / self.gamma if not np.isnan(signed_ratio) else float("nan")

        row = [
            self._sync_count, training_step,
            m["mean_td1"], m["std_td1"],
            m["mean_td2"], m["std_td2"],
            signed_ratio,
            m["abs_be_q1"], m["abs_be_q2"],
            bias_ema,
            self.gamma,
            ratio_vs_gamma,
        ]
        self.records.append(row)
        self._writer.writerow(row)
        self._f.flush()

    def close(self):
        self._f.close()

    def summary(self):
        if not self.records:
            return "No sync records."
        # columns: sync_step, training_steps, mean_td1, std_td1, mean_td2, std_td2,
        #          signed_ratio, abs_be_q1, abs_be_q2, bias_ema, gamma, ratio_vs_gamma
        signed_ratios = [r[6] for r in self.records if not np.isnan(r[6])]
        ratio_vs_gammas = [r[11] for r in self.records if not np.isnan(r[11])]
        mean_td1s = [r[2] for r in self.records]
        mean_td2s = [r[4] for r in self.records]
        return (
            f"Syncs: {len(self.records)}\n"
            f"  mean_td1 (E[Q1-y])  mean={np.mean(mean_td1s):.5f}  "
            f"(+= overestimation, bias_ema tracks this)\n"
            f"  mean_td2 (E[Q2-y])  mean={np.mean(mean_td2s):.5f}  "
            f"(predicted ≈ gamma * mean_td1)\n"
            f"  signed_ratio        mean={np.mean(signed_ratios):.4f} ± {np.std(signed_ratios):.4f}\n"
            f"  gamma prediction    {self.gamma:.4f}\n"
            f"  ratio/gamma         mean={np.mean(ratio_vs_gammas):.4f}  "
            f"(1.0 = theorem exactly holds)\n"
            f"  |signed_ratio|<1    {np.mean([abs(r)<1 for r in signed_ratios]):.1%}"
        )
