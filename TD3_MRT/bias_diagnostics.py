"""
Empirical verification of ART mean-Bellman-error reduction theorem (TD3 version).

Theoretical prediction:
  After each sync (bias_qf → qf, qf_target), measured on a held-out batch:
      mean |Q2(s,a) - y_standard| ≈ gamma * mean |Q1(s,a) - y_standard|
  i.e., be_ratio = be_Q2 / be_Q1 ≈ gamma = 0.99

Proposition proof sketch (in comments below):
  If F is closed under constant shifts and bias = E[epsilon_Q1] exactly, then
  Q2* = Q1* - bias * 1, so epsilon_Q2 = epsilon_Q1 - (1-gamma)*bias*1,
  thus mean(epsilon_Q2) = gamma * mean(epsilon_Q1).
"""

import csv
import os
import numpy as np
import torch


class TD3ARTDiagnostics:
    def __init__(self, trainer, log_path: str):
        self.trainer = trainer
        self.records = []
        self._sync_count = 0

        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
        self._f = open(log_path, "w", newline="")
        self._writer = csv.writer(self._f)
        self._writer.writerow([
            "sync_step", "training_steps",
            "mean_td1", "std_td1",
            "mean_td2", "std_td2",
            "signed_ratio",
            "abs_be_q1", "abs_be_q2",
            "bias_ema",
            "gamma",
            "ratio_vs_gamma",
        ])
        self._f.flush()

    def measure(self, replay_buffer, batch_size=1024):
        trainer = self.trainer
        if not trainer.mrt:
            return None

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * trainer.policy_noise).clamp(
                -trainer.noise_clip, trainer.noise_clip)
            next_action = (trainer.actor_target(next_state) + noise).clamp(
                -trainer.max_action, trainer.max_action)
            nq1, nq2 = trainer.qf_target(next_state, next_action)
            min_nq = torch.min(nq1, nq2)
            y = reward + not_done * trainer.discount * min_nq + trainer.ptbs  # (B, n_ptb)

            # Signed TD errors (theory is about signed mean, not absolute)
            q1_vals, _ = trainer.qf(state, action)
            td1 = (q1_vals - y).mean(dim=1)           # mean over ptb → (B,) signed

            # bias_qf only exists in the separate-network variant; scalar MRT has no bias_qf
            if not hasattr(trainer, 'bias_qf'):
                return None
            bq1_vals, _ = trainer.bias_qf(state, action)
            td2 = (bq1_vals - y).mean(dim=1)           # bias_net signed TD error

        return {
            "mean_td1":  float(td1.mean()),
            "std_td1":   float(td1.std()),
            "mean_td2":  float(td2.mean()),
            "std_td2":   float(td2.std()),
            "abs_be_q1": float(td1.abs().mean()),
            "abs_be_q2": float(td2.abs().mean()),
        }

    def record_sync(self, training_steps: int, replay_buffer, batch_size=1024):
        self._sync_count += 1
        m = self.measure(replay_buffer, batch_size)
        if m is None:
            return

        bias_ema = float(self.trainer.bias_ema)
        gamma = float(self.trainer.discount)
        signed_ratio = m["mean_td2"] / (m["mean_td1"] + 1e-8) if abs(m["mean_td1"]) > 1e-6 else float("nan")
        ratio_vs_gamma = signed_ratio / gamma if not (isinstance(signed_ratio, float) and np.isnan(signed_ratio)) else float("nan")

        row = [
            self._sync_count, training_steps,
            m["mean_td1"], m["std_td1"],
            m["mean_td2"], m["std_td2"],
            signed_ratio,
            m["abs_be_q1"], m["abs_be_q2"],
            bias_ema,
            gamma,
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
        signed_ratios = [r[6] for r in self.records if not (isinstance(r[6], float) and np.isnan(r[6]))]
        ratio_vs_gammas = [r[11] for r in self.records if not (isinstance(r[11], float) and np.isnan(r[11]))]
        mean_td1s = [r[2] for r in self.records]
        mean_td2s = [r[4] for r in self.records]
        gamma = self.records[0][10]
        return (
            f"Syncs recorded: {len(self.records)}\n"
            f"  mean_td1 (E[Q1-y])  mean={np.mean(mean_td1s):.5f}  (bias_ema tracks this)\n"
            f"  mean_td2 (E[Q2-y])  mean={np.mean(mean_td2s):.5f}  (predicted ≈ gamma*mean_td1)\n"
            f"  signed_ratio        mean={np.mean(signed_ratios):.4f} ± {np.std(signed_ratios):.4f}\n"
            f"  gamma               {gamma:.4f}\n"
            f"  ratio/gamma         mean={np.mean(ratio_vs_gammas):.4f}  (1.0 = theorem holds)\n"
            f"  |signed_ratio|<1    {np.mean([abs(r) < 1 for r in signed_ratios]):.1%} of syncs"
        )
