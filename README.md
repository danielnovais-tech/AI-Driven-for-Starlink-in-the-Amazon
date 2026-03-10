# AI-Driven Beamforming for Starlink in the Amazon

AI algorithms, especially Deep Reinforcement Learning (DRL) and hybrid DNNs,
optimize Starlink's real-time beamforming over the Amazon rainforest.
The framework mitigates tropical attenuation (rain, foliage, humidity) via
adaptive beam steering, power control, MCS selection and resource-block
allocation, ensuring resilient connectivity for communities, telemedicine
and environmental monitoring.

---

## Architecture overview

```
src/
├── data/
│   ├── telemetry_dataset.py   # PyTorch Dataset for Starlink CSV telemetry
│   └── radar_dataset.py       # PyTorch Dataset for meteorological HDF5 radar
├── channel/
│   └── rain_attenuation.py    # ITU-R P.838-3 rain attenuation + ChannelModel
├── envs/
│   ├── leo_beamforming_env.py # Gymnasium MDP environment (online simulation)
│   └── offline_env.py         # Offline environment (replays recorded logs)
├── agents/
│   ├── networks.py            # BeamformingNetwork (actor-critic) + DQNNetwork
│   ├── dqn_agent.py           # DQN with experience replay and target network
│   ├── ppo_agent.py           # PPO with GAE and clipped surrogate objective
│   └── a3c_agent.py           # A3C with asynchronous worker threads
├── gnn/
│   └── cognitive_beam_gnn.py  # Graph Attention Network for multi-sat coord.
├── inference/
│   └── online_controller.py   # Real-time beam controller (<=500 ms loop)
└── utils/
    └── evaluation.py          # Throughput / latency / outage metrics
tests/                         # pytest suite (64 tests, all passing)
```

---

## Key modules

### 1. Data pipeline
| Class | File | Description |
|---|---|---|
| `TelemetryDataset` | `src/data/telemetry_dataset.py` | Sliding-window Dataset over Starlink CSV telemetry (SNR, RSSI, orbital position). |
| `RadarDataset` | `src/data/radar_dataset.py` | Grid Dataset for CPTEC/INPE or GPM HDF5 rain-rate maps. |

### 2. Channel model - ITU-R P.838-3
`src/channel/rain_attenuation.py` implements:
- `rain_specific_attenuation(R, f, pol)` - specific attenuation gamma (dB/km).
- `slant_path_attenuation(R, elevation, ...)` - total slant-path rain loss (dB).
- `ChannelModel` - full link-budget model (FSPL + rain + foliage).

### 3. MDP environments
| Class | Description |
|---|---|
| `LEOBeamformingEnv` | Online Gymnasium env; state = [SNR, RSSI, sat_pos x3, rain, LAI], action = [delta_phase, delta_power, MCS, RBs]. |
| `OfflineLEOEnv` | Replay environment backed by HDF5 logs for offline DRL training. |

### 4. DRL agents
| Agent | Algorithm | Action space |
|---|---|---|
| `DQNAgent` | Deep Q-Network (Mnih et al., 2015) | Discrete |
| `PPOAgent` | Proximal Policy Optimisation (Schulman et al., 2017) | Continuous |
| `A3CWorker` / `run_a3c` | Async Advantage Actor-Critic (Mnih et al., 2016) | Continuous |

### 5. Cognitive Beamforming GNN
`CognitiveBeamGNN` (Graph Attention Network) coordinates multiple satellites
by propagating channel quality, traffic load and interference information
across a heterogeneous graph of satellite and ground-station nodes.
Requires `torch-geometric`.

### 6. Online inference
`OnlineBeamController` polls sensors every <=500 ms, normalises the state,
calls the trained agent deterministically and issues beam-steering commands
via `apply_beam_steering()`.

---

## Installation

```bash
pip install torch numpy pandas gymnasium h5py
# Optional - for GNN module:
pip install torch-geometric
```

---

## Running the tests

```bash
pytest tests/ -v
```

---

## References

- ITU-R P.838-3: Specific attenuation model for rain.
- ITU-R P.618-13: Earth-space propagation prediction methods.
- Mnih et al. (2015): Human-level control through deep RL (DQN).
- Schulman et al. (2017): Proximal Policy Optimization Algorithms.
- Mnih et al. (2016): Asynchronous Methods for Deep RL (A3C).
- Velickovic et al. (2018): Graph Attention Networks.
- Data sources: CPTEC/INPE radar, NASA GPM, INPE PRODES, MODIS LAI.
