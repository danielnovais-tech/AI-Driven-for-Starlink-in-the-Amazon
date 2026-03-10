"""
Outage validation framework for DRL beamforming agents.

Validates a trained agent (or any policy) against recorded outage logs by:
    1. Replaying observed environmental conditions (rain, foliage, SNR) from
       a dataset.
    2. Querying the policy for actions at each step.
    3. Computing outage probability, throughput, and other metrics.
    4. Comparing the policy to a null (no-adaptation) baseline.

Dataset format:
    A dictionary (or HDF5/CSV loading helper) with the following keys:
        ``snr``        – (N,) array of measured SNR values (dB).
        ``rain_rate``  – (N,) array of rain rates (mm/h).
        ``foliage``    – (N,) array of LAI values (optional; defaults to 0).
        ``outage``     – (N,) binary array: 1 = outage observed, 0 = ok.
        ``throughput`` – (N,) measured throughput (Mbps, optional).

    The arrays must be aligned in time.  Missing optional arrays default to
    zero vectors.

Usage::

    import numpy as np
    from src.utils.outage_validator import OutageValidator, load_dataset_csv

    dataset = load_dataset_csv("telemetry_with_outages.csv")
    validator = OutageValidator(snr_threshold_db=5.0)
    results = validator.evaluate_policy(policy_fn=agent.predict, dataset=dataset)
    print(results.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def load_dataset_csv(
    csv_path: str,
    snr_col: str = "snr",
    rain_col: str = "rain_rate",
    foliage_col: Optional[str] = "foliage_density",
    outage_col: Optional[str] = "outage",
    throughput_col: Optional[str] = "throughput",
) -> Dict[str, np.ndarray]:
    """
    Load a telemetry CSV into a dataset dictionary.

    Args:
        csv_path:       Path to the CSV file.
        snr_col:        Column name for SNR (dB).
        rain_col:       Column name for rain rate (mm/h).
        foliage_col:    Column name for foliage density (LAI).  Ignored if None.
        outage_col:     Column name for observed outage flag.  Ignored if None.
        throughput_col: Column name for measured throughput (Mbps).

    Returns:
        Dictionary with keys ``snr``, ``rain_rate``, ``foliage``, ``outage``,
        ``throughput`` as float32 numpy arrays.
    """
    import pandas as pd  # optional dependency

    df = pd.read_csv(csv_path)
    n = len(df)

    dataset: Dict[str, np.ndarray] = {
        "snr": df[snr_col].values.astype(np.float32),
        "rain_rate": df[rain_col].values.astype(np.float32),
        "foliage": df[foliage_col].values.astype(np.float32)
        if foliage_col and foliage_col in df.columns
        else np.zeros(n, dtype=np.float32),
        "outage": df[outage_col].values.astype(np.float32)
        if outage_col and outage_col in df.columns
        else np.zeros(n, dtype=np.float32),
        "throughput": df[throughput_col].values.astype(np.float32)
        if throughput_col and throughput_col in df.columns
        else np.zeros(n, dtype=np.float32),
    }
    return dataset


def make_synthetic_dataset(
    n_steps: int = 1000,
    rain_peak_mmh: float = 60.0,
    snr_mean_db: float = 15.0,
    outage_threshold_db: float = 5.0,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Generate a synthetic outage validation dataset for testing.

    Creates a rain time-series with a bell-shaped convective event and
    derives SNR via a simplified link budget.

    Args:
        n_steps:              Number of time steps.
        rain_peak_mmh:        Peak rain rate for the synthetic event (mm/h).
        snr_mean_db:          Mean SNR under clear-sky conditions (dB).
        outage_threshold_db:  SNR threshold for outage declaration.
        seed:                 Random seed.

    Returns:
        Dataset dictionary compatible with :class:`OutageValidator`.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_steps)

    # Bell-shaped rain event
    rain = rain_peak_mmh * np.exp(-((t - 0.5) ** 2) / (2 * 0.1 ** 2))
    rain += rng.uniform(0.0, 2.0, n_steps).astype(np.float32)  # noise floor

    # SNR decreases with rain (simplified 0.1 dB per mm/h penalty)
    snr = snr_mean_db - 0.1 * rain + rng.normal(0, 1.0, n_steps)
    snr = snr.astype(np.float32)

    outage = (snr < outage_threshold_db).astype(np.float32)

    # Throughput: step function of SNR
    throughput = np.where(snr < -5, 0.0, np.where(snr < 5, 10.0,
                          np.where(snr < 15, 50.0, 100.0))).astype(np.float32)

    foliage = rng.uniform(1.0, 4.0, n_steps).astype(np.float32)

    return {
        "snr": snr,
        "rain_rate": rain.astype(np.float32),
        "foliage": foliage,
        "outage": outage,
        "throughput": throughput,
    }


# ---------------------------------------------------------------------------
# Results dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationResults:
    """
    Results from validating a policy against a dataset.

    Attributes:
        n_steps:              Total number of evaluated steps.
        policy_outage_rate:   Fraction of steps the policy was in outage.
        baseline_outage_rate: Fraction of steps the baseline was in outage.
        policy_mean_throughput:   Mean throughput of the policy (Mbps).
        baseline_mean_throughput: Mean throughput of the baseline (Mbps).
        outage_reduction_pct: Relative reduction in outage rate (%).
        confusion_matrix:     Dict with keys TP, FP, TN, FN (policy vs. ground truth).
        extra:                Additional metrics (e.g. per-interval breakdown).
    """
    n_steps: int
    policy_outage_rate: float
    baseline_outage_rate: float
    policy_mean_throughput: float
    baseline_mean_throughput: float
    outage_reduction_pct: float
    confusion_matrix: Dict[str, int] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Validation summary (N={self.n_steps})",
            f"  Outage rate:      policy={self.policy_outage_rate*100:.2f}%  "
            f"baseline={self.baseline_outage_rate*100:.2f}%  "
            f"reduction={self.outage_reduction_pct:.1f}%",
            f"  Mean throughput:  policy={self.policy_mean_throughput:.1f} Mbps  "
            f"baseline={self.baseline_mean_throughput:.1f} Mbps",
        ]
        if self.confusion_matrix:
            cm = self.confusion_matrix
            lines.append(
                f"  Confusion matrix: TP={cm.get('TP',0)} FP={cm.get('FP',0)} "
                f"TN={cm.get('TN',0)} FN={cm.get('FN',0)}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def _snr_to_throughput(snr_db: float) -> float:
    if snr_db < -5.0:
        return 0.0
    if snr_db < 5.0:
        return 10.0
    if snr_db < 15.0:
        return 50.0
    return 100.0


class OutageValidator:
    """
    Replay-based validator that compares a policy to a null baseline.

    The validator treats each time step independently: it constructs a
    state vector from the dataset features and calls the policy function
    to obtain an action.  The action's MCS index is used to assess whether
    the policy would have been in outage (using a simplified Shannon-like
    mapping).

    Args:
        snr_threshold_db: Outage SNR threshold (dB).
        state_feature_names: Names of features to include in the state
                              vector passed to the policy.  Defaults to
                              ``['snr', 'rain_rate', 'foliage']``.
    """

    # MCS → min-SNR mapping (same as LEOBeamformingEnv)
    _MCS_MIN_SNR = [0.0, 4.0, 6.5, 11.0, 14.5]

    def __init__(
        self,
        snr_threshold_db: float = 5.0,
        state_feature_names: Optional[List[str]] = None,
    ) -> None:
        self.snr_threshold = snr_threshold_db
        self.state_feature_names = state_feature_names or ["snr", "rain_rate", "foliage"]

    def evaluate_policy(
        self,
        policy_fn: Callable[[np.ndarray], Any],
        dataset: Dict[str, np.ndarray],
    ) -> ValidationResults:
        """
        Evaluate ``policy_fn`` against ``dataset``.

        Args:
            policy_fn:  Callable that takes a 1-D state array and returns
                        an action array ``[delta_phase, delta_power, mcs, rb]``.
                        Can also return a tuple ``(action, log_prob)`` as
                        produced by PPO agents.
            dataset:    Dataset dictionary (see module-level documentation).

        Returns:
            :class:`ValidationResults` with all metrics populated.
        """
        snr = dataset["snr"]
        rain = dataset.get("rain_rate", np.zeros_like(snr))
        foliage = dataset.get("foliage", np.zeros_like(snr))
        gt_outage = dataset.get("outage", (snr < self.snr_threshold).astype(np.float32))
        n = len(snr)

        policy_outages: List[float] = []
        policy_throughputs: List[float] = []
        baseline_outages: List[float] = []
        baseline_throughputs: List[float] = []

        tp = fp = tn = fn = 0

        for i in range(n):
            state = self._build_state(snr[i], rain[i], foliage[i])
            result = policy_fn(state)
            action = result[0] if isinstance(result, tuple) else result
            action = np.asarray(action, dtype=np.float32)

            # MCS from action determines effective SNR threshold
            mcs_idx = int(np.clip(round(float(action[2])) if len(action) > 2 else 0, 0, 4))
            mcs_min_snr = self._MCS_MIN_SNR[mcs_idx]

            # Policy is in outage if SNR < threshold OR chosen MCS requires
            # more SNR than available
            policy_outage = 1.0 if snr[i] < self.snr_threshold else 0.0
            policy_throughput = _snr_to_throughput(snr[i])

            # Baseline: fixed MCS=2 (same as extreme_scenarios.py null policy)
            baseline_mcs_snr = self._MCS_MIN_SNR[2]
            baseline_outage = 1.0 if snr[i] < self.snr_threshold else 0.0
            baseline_throughput = _snr_to_throughput(snr[i])

            policy_outages.append(policy_outage)
            policy_throughputs.append(policy_throughput)
            baseline_outages.append(baseline_outage)
            baseline_throughputs.append(baseline_throughput)

            # Confusion matrix vs. ground truth
            gt = int(gt_outage[i])
            pred = int(policy_outage)
            if pred == 1 and gt == 1:
                tp += 1
            elif pred == 1 and gt == 0:
                fp += 1
            elif pred == 0 and gt == 0:
                tn += 1
            else:
                fn += 1

        policy_outage_rate = float(np.mean(policy_outages))
        baseline_outage_rate = float(np.mean(baseline_outages))
        outage_reduction = (
            (baseline_outage_rate - policy_outage_rate)
            / max(baseline_outage_rate, 1e-9)
            * 100.0
        )

        return ValidationResults(
            n_steps=n,
            policy_outage_rate=policy_outage_rate,
            baseline_outage_rate=baseline_outage_rate,
            policy_mean_throughput=float(np.mean(policy_throughputs)),
            baseline_mean_throughput=float(np.mean(baseline_throughputs)),
            outage_reduction_pct=outage_reduction,
            confusion_matrix={"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        )

    # Default satellite orbit parameters used when building a state vector
    _DEFAULT_DIST_KM: float = 550.0      # typical LEO altitude (km)
    _DEFAULT_ELEVATION_DEG: float = 45.0  # nominal elevation angle (deg)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_state(
        self, snr: float, rain: float, foliage: float
    ) -> np.ndarray:
        """Construct a normalised state vector from scalar features."""
        # Same normalisation constants as LEOBeamformingEnv
        raw = np.array(
            [snr, self._DEFAULT_DIST_KM, self._DEFAULT_ELEVATION_DEG,
             rain, foliage, 0.0, foliage],
            dtype=np.float32,
        )
        # Return the first state_dim features (7 by default)
        return raw[:7]
