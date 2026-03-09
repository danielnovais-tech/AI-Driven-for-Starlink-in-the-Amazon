"""
Gradient-based explainability for DRL beamforming agents.

Provides saliency maps and feature attribution techniques that explain
*why* the agent chose a particular action, without requiring external
libraries like SHAP or LIME.  All methods are based on standard
backpropagation through the policy network.

Supported methods:
    - ``vanilla_saliency``  – gradient of the action output w.r.t. the
                              input state (Simonyan et al., 2014).
    - ``integrated_gradients`` – path integral of gradients from a
                                  baseline to the actual input
                                  (Sundararajan et al., 2017).
    - ``smooth_grad``       – averaged saliency over noisy input copies
                              (Smilkov et al., 2017).

All methods return a numpy array of shape ``(state_dim,)`` representing
the importance of each state feature for the agent's action.

References:
    Simonyan et al. (2014) – "Deep Inside Convolutional Networks."
    Sundararajan et al. (2017) – "Axiomatic Attribution for DNNs."
    Smilkov et al. (2017) – "SmoothGrad: removing noise by adding noise."
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def vanilla_saliency(
    model: nn.Module,
    state: np.ndarray,
    action_idx: Optional[int] = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute the gradient of the policy output w.r.t. the input state.

    For a continuous actor-critic network the gradient is taken w.r.t. the
    action mean output.  If ``action_idx`` is provided, only that output
    dimension is differentiated.

    Args:
        model:       Actor-critic network with signature ``forward(x) ->
                     (action_mean, value)``.
        state:       1-D state array (unnormalised or normalised).
        action_idx:  Index of the action dimension to differentiate.  If
                     None, the sum over all action dimensions is used.
        device:      Torch device string.

    Returns:
        Saliency array of shape ``(state_dim,)`` (absolute gradients).
    """
    model.eval()
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    state_t.requires_grad_(True)

    action_mean, _value = model(state_t)
    if action_idx is not None:
        output = action_mean[0, action_idx]
    else:
        output = action_mean.sum()

    model.zero_grad()
    output.backward()

    grad = state_t.grad.detach().cpu().numpy().flatten()
    return np.abs(grad)


def integrated_gradients(
    model: nn.Module,
    state: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    n_steps: int = 50,
    action_idx: Optional[int] = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute Integrated Gradients (IG) attributions.

    Approximates the path integral of gradients from a baseline (typically
    the zero vector) to the actual input:

        IG_i(x) = (x_i - x'_i) * int_0^1 dF/dx_i(x' + alpha*(x-x')) dalpha

    Args:
        model:       Policy network (same interface as :func:`vanilla_saliency`).
        state:       1-D input state array.
        baseline:    1-D baseline array; defaults to zeros.
        n_steps:     Number of Riemann approximation steps.
        action_idx:  Action dimension to attribute (or None for sum).
        device:      Torch device string.

    Returns:
        Attribution array of shape ``(state_dim,)`` (may be positive or
        negative; positive ⇒ feature increases the selected action output).
    """
    if baseline is None:
        baseline = np.zeros_like(state, dtype=np.float32)

    state = np.asarray(state, dtype=np.float32)
    baseline = np.asarray(baseline, dtype=np.float32)
    delta = state - baseline

    grads_sum = np.zeros_like(state, dtype=np.float32)

    model.eval()
    for k in range(1, n_steps + 1):
        alpha = k / n_steps
        interp = baseline + alpha * delta
        interp_t = torch.FloatTensor(interp).unsqueeze(0).to(device)
        interp_t.requires_grad_(True)

        action_mean, _ = model(interp_t)
        if action_idx is not None:
            output = action_mean[0, action_idx]
        else:
            output = action_mean.sum()

        model.zero_grad()
        output.backward()
        grads_sum += interp_t.grad.detach().cpu().numpy().flatten()

    # Approximate integral via trapezoidal rule
    attributions = delta * (grads_sum / n_steps)
    return attributions


def smooth_grad(
    model: nn.Module,
    state: np.ndarray,
    n_samples: int = 30,
    noise_std: float = 0.1,
    action_idx: Optional[int] = None,
    device: str = "cpu",
    seed: int = 0,
) -> np.ndarray:
    """
    Compute SmoothGrad attributions by averaging saliency over noisy copies.

    Args:
        model:       Policy network.
        state:       1-D state array.
        n_samples:   Number of noisy copies to average over.
        noise_std:   Standard deviation of added Gaussian noise.
        action_idx:  Action dimension to attribute (or None for sum).
        device:      Torch device string.
        seed:        Random seed for reproducibility.

    Returns:
        Smoothed saliency array of shape ``(state_dim,)``.
    """
    rng = np.random.default_rng(seed)
    saliency_sum = np.zeros_like(state, dtype=np.float32)

    for _ in range(n_samples):
        noise = rng.normal(0.0, noise_std, state.shape).astype(np.float32)
        noisy_state = state + noise
        sal = vanilla_saliency(model, noisy_state, action_idx=action_idx, device=device)
        saliency_sum += sal

    return saliency_sum / n_samples


def feature_importance_summary(
    model: nn.Module,
    states: np.ndarray,
    feature_names: Optional[list] = None,
    method: str = "smooth_grad",
    device: str = "cpu",
) -> dict:
    """
    Compute mean feature importances over a batch of states.

    Args:
        model:         Policy network.
        states:        2-D array of shape (N, state_dim).
        feature_names: Optional list of feature name strings.  If None,
                       names default to ``['f_0', 'f_1', ...]``.
        method:        Attribution method: ``'vanilla'``, ``'ig'``, or
                       ``'smooth_grad'`` (default).
        device:        Torch device string.

    Returns:
        Dictionary mapping feature name → mean absolute attribution.
    """
    states = np.asarray(states, dtype=np.float32)
    n, state_dim = states.shape

    if feature_names is None:
        feature_names = [f"f_{i}" for i in range(state_dim)]

    aggregated = np.zeros(state_dim, dtype=np.float32)
    for i in range(n):
        if method == "ig":
            attr = np.abs(integrated_gradients(model, states[i], device=device))
        elif method == "vanilla":
            attr = vanilla_saliency(model, states[i], device=device)
        else:
            attr = smooth_grad(model, states[i], device=device)
        aggregated += attr

    mean_importances = aggregated / n
    return dict(zip(feature_names, mean_importances.tolist()))
