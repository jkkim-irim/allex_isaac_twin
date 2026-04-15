# ALLEX IsaacTwin

Real2Sim digital twin extension for ALLEX humanoid robot. Subscribes to joint position data published via ROS2 and mirrors the real robot's motion in Isaac Sim in real time.

## Requirements

- NVIDIA Isaac Sim 5.1.0+
- ROS2 (uses Isaac Sim built-in bridge — no separate ROS2 install needed)

> **Do NOT** `source /opt/ros/*/setup.bash` before launching Isaac Sim.  
> System ROS2 (Python 3.12) conflicts with Isaac Sim (Python 3.11) and causes a core dump.

## Installation

1. Copy this folder into Isaac Sim's `extsUser/` directory:
   ```
   <isaac-sim-root>/extsUser/ALLEX_IsaacTwin/
   ```
2. Launch Isaac Sim → `Window > Extensions` → search `ALLEX` → enable

## Usage

1. **LOAD** — Create stage + load ALLEX robot
2. **RUN** — Start simulation
3. **Domain ID** — Set ROS2 Domain ID (default: 77)
4. **Initialize ROS2** — Create ROS2 node with specified Domain ID
5. **Subscriber** — Start subscribing to 28 joint topics (14 current + 14 desired)
6. **Current | Desired** — Toggle control mode
7. **Force Visualizer** — Toggle force visualization on L/R hand links (12 prims)

## ROS2 Topics

Default Domain ID: `77` | RMW: `rmw_cyclonedds_cpp`

Both are configurable via environment variables (`ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION`) or the UI.

### Joint Position Topics

14 groups × 2 modes = 28 topics. The robot mirrors these in real time.

| Group | Topic | Joints |
|-------|-------|--------|
| Right Arm | `/robot_outbound_data/Arm_R_theOne/{mode}` | 7 |
| Left Arm | `/robot_outbound_data/Arm_L_theOne/{mode}` | 7 |
| Right Hand (×5) | `/robot_outbound_data/Hand_R_{finger}_wir/{mode}` | 3 each |
| Left Hand (×5) | `/robot_outbound_data/Hand_L_{finger}_wir/{mode}` | 3 each |
| Waist | `/robot_outbound_data/theOne_waist/{mode}` | 2 |
| Neck | `/robot_outbound_data/theOne_neck/{mode}` | 2 |

- `{mode}`: `joint_positions_deg` (current) or `joint_ang_target_deg` (desired)
- `{finger}`: `thumb`, `index`, `middle`, `ring`, `little`
- Message type: `std_msgs/msg/Float64MultiArray` (degree)

### Joint Torque Topics (planned)

14 groups × 2 directions = 28 topics. Will be used for force visualization.

| Direction | Topic Pattern |
|-----------|--------------|
| Inbound | `/robot_inbound/{group}/joint_torque` |
| Outbound | `/robot_outbound_data/{group}/joint_torque` |

Torque data will be converted to force vectors and rendered via `force_viz` prims on each hand link.

## Directory Structure

```
ALLEX_IsaacTwin/
├── config/extension.toml        # Extension config
├── asset/
│   ├── ALLEX/ALLEX.usd          # Robot USD model
│   └── utils/                   # force_viz.usd, torque_viz.usd
├── src/
│   ├── extension.py             # Extension entry point
│   ├── scenario.py              # Real2Sim orchestrator
│   ├── ui_builder.py            # UI entry point (delegates to controllers)
│   ├── config/                  # ROS2, UI, joint config
│   ├── core/                    # Asset loading, joint control, sim loop
│   ├── ros2/                    # ROS2 node and communication
│   ├── ui_utils/                # UI controllers (world, ros2, visualizer)
│   └── utils/                   # Constants
└── temp/                        # Archived code (visualization, sensor)
```

## Robot Specs

- Total joints: 59 (49 active + 10 coupled)
- Coupled joints: auto-calculated from master joint ratio (`joint_config.json`)
- Spawn position: `(0, 0, 0.685)`
