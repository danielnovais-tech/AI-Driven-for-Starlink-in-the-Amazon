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


# ---------------------------------------------------------------------------
# GNN node importance (GradCAM-style)
# ---------------------------------------------------------------------------

def gnn_node_importance(
    agent,
    graph_obs,
    node_type: str = "sat",
    feature_names: Optional[list] = None,
    device: str = "cpu",
) -> dict:
    """
    Compute per-node and per-feature importance scores for a GNN agent.

    Uses input-gradient saliency (∂output/∂node_features) to score each
    satellite node and each feature dimension.  The method is analogous to
    ``vanilla_saliency`` but operates on the heterogeneous graph input used
    by :class:`~agents.gnn_ppo_agent.GNNPPOAgent`.

    Requirements:
        ``torch_geometric`` must be installed.

    Args:
        agent:         A :class:`~agents.gnn_ppo_agent.GNNPPOAgent` whose
                       ``net`` has a ``forward_actor`` method or standard
                       ``forward`` returning ``(action_logits, value)``.
        graph_obs:     A ``HeteroData`` graph observation as produced by
                       :class:`~inference.online_controller.GNNBeamController`.
        node_type:     Node type whose features are differentiated
                       (default ``"sat"``).
        feature_names: Optional list of feature names for the node feature
                       vector (e.g. ``["snr", "distance", "elevation",
                       "rain_rate"]``).  Defaults to ``["f_0", "f_1", ...]``.
        device:        Torch device string.

    Returns:
        Dictionary with keys:
            ``node_scores``    – list of per-node importance floats (L2 norm
                                 of the gradient w.r.t. each node's features).
            ``feature_scores`` – dict mapping feature name → mean absolute
                                 gradient across all nodes.
            ``top_node``       – index of the most important satellite node.
            ``top_features``   – list of (feature_name, score) sorted by score
                                 in descending order.
    """
    try:
        import torch as _torch
    except ImportError as exc:
        raise ImportError("gnn_node_importance requires torch.") from exc

    net = agent.net
    net.eval()

    # Deep-copy the graph so the differentiable x does not alias the original
    # object's tensors, preventing unintended side effects on shared data.
    import copy
    data = copy.deepcopy(graph_obs)

    # Enable gradient tracking for node features
    x = data[node_type].x.to(device).detach().clone().requires_grad_(True)
    # The differentiable node features are already set in data above

    # Forward pass
    try:
        logits, _value = net(data)
    except Exception:
        # Fallback: some GNN nets expose get_action internals differently
        out = net(data)
        logits = out[0] if isinstance(out, tuple) else out

    # Differentiate w.r.t. selected node features
    output = logits.sum()
    net.zero_grad()
    if x.grad is not None:
        x.grad.zero_()
    output.backward()

    grad = x.grad.detach().cpu().numpy()  # shape: (n_nodes, n_features)
    n_nodes, n_features = grad.shape

    # Per-node importance: L2 norm of gradient across features
    node_scores = [float(np.linalg.norm(grad[i])) for i in range(n_nodes)]
    top_node = int(np.argmax(node_scores))

    # Per-feature importance: mean absolute gradient across nodes
    if feature_names is None:
        feature_names = [f"f_{i}" for i in range(n_features)]
    feat_scores = {
        feature_names[j]: float(np.mean(np.abs(grad[:, j])))
        for j in range(min(n_features, len(feature_names)))
    }
    top_features = sorted(feat_scores.items(), key=lambda kv: kv[1], reverse=True)

    return {
        "node_scores": node_scores,
        "feature_scores": feat_scores,
        "top_node": top_node,
        "top_features": top_features,
    }


# ---------------------------------------------------------------------------
# DecisionExplainer: integrates explainability into the controller loop
# ---------------------------------------------------------------------------

class DecisionExplainer:
    """
    Wraps explainability calls and formats results for structured logging.

    Designed to be embedded in :class:`~inference.online_controller.OnlineBeamController`
    (or any subclass) and called once per ``step()`` to produce an
    ``explanation`` dict that is included in the step result and emitted
    to the structured log.

    For flat-state agents (PPO, DQN) the explanation is a feature-importance
    summary produced by :func:`smooth_grad`.  For GNN agents it additionally
    includes per-node scores produced by :func:`gnn_node_importance`.

    Args:
        agent:         The DRL agent (must have a ``net`` attribute for
                       flat-state agents; must have ``get_action`` for GNN).
        feature_names: State feature names for flat-state attribution.
        method:        Attribution method for flat-state agents:
                       ``"vanilla"``, ``"ig"``, or ``"smooth_grad"``
                       (default).
        gnn_node_feat_names: Feature names for GNN node features.
        device:        Torch device string.
        enabled:       If ``False``, :meth:`explain` returns an empty dict
                       immediately (zero overhead).
    """

    def __init__(
        self,
        agent,
        feature_names: Optional[list] = None,
        method: str = "smooth_grad",
        gnn_node_feat_names: Optional[list] = None,
        device: str = "cpu",
        enabled: bool = True,
    ) -> None:
        self.agent = agent
        self.feature_names = feature_names
        self.method = method
        self.gnn_node_feat_names = gnn_node_feat_names or ["snr", "distance", "elevation", "rain_rate"]
        self.device = device
        self.enabled = enabled

    def explain(self, state_or_graph, action) -> dict:
        """
        Produce an explanation for the given state/graph and action.

        Args:
            state_or_graph: Either a 1-D numpy state array (flat agents) or
                            a ``HeteroData`` graph (GNN agents).
            action:         The action selected by the agent (for context).

        Returns:
            Explanation dictionary:
                ``method``          – attribution method used.
                ``feature_scores``  – feature name → importance score.
                ``top_features``    – top-3 (name, score) pairs.
                ``action``          – the action that was explained.
                For GNN agents additionally:
                ``node_scores``     – per-satellite-node importance.
                ``top_node``        – index of most important satellite.
        """
        if not self.enabled:
            return {}

        explanation: dict = {"method": self.method, "action": action}

        # Detect GNN agent by checking if the input is a HeteroData graph
        is_graph = _is_heterodata(state_or_graph)

        if is_graph:
            try:
                result = gnn_node_importance(
                    self.agent,
                    state_or_graph,
                    node_type="sat",
                    feature_names=self.gnn_node_feat_names,
                    device=self.device,
                )
                explanation["method"] = "gnn_gradient"
                explanation.update(result)
                explanation["top_features"] = [
                    (k, v) for k, v in result["top_features"][:3]
                ]
            except Exception as exc:  # noqa: BLE001
                explanation["error"] = str(exc)
        else:
            try:
                net = getattr(self.agent, "net", None)
                if net is not None:
                    state = np.asarray(state_or_graph, dtype=np.float32)
                    if self.method == "ig":
                        attr = np.abs(integrated_gradients(net, state, device=self.device))
                    elif self.method == "vanilla":
                        attr = vanilla_saliency(net, state, device=self.device)
                    else:
                        attr = smooth_grad(net, state, device=self.device)

                    n_feat = len(attr)
                    names = self.feature_names or [f"f_{i}" for i in range(n_feat)]
                    feat_scores = {names[i]: float(attr[i]) for i in range(min(n_feat, len(names)))}
                    top_feats = sorted(feat_scores.items(), key=lambda kv: kv[1], reverse=True)
                    explanation["feature_scores"] = feat_scores
                    explanation["top_features"] = top_feats[:3]
            except Exception as exc:  # noqa: BLE001
                explanation["error"] = str(exc)

        return explanation


def _is_heterodata(obj) -> bool:
    """Return True if ``obj`` is a torch_geometric HeteroData instance."""
    try:
        from torch_geometric.data import HeteroData
        return isinstance(obj, HeteroData)
    except ImportError:
        return False
