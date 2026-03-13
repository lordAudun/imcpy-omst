## Running the dive-follow-pipe maneuver

The `dive_follow_pipe_maneuver.py` script sends a dive-and-follow-pipe mission to a LAUV vehicle via DUNE/Neptus.

### Prerequisites

- A running LAUV simulator or real vehicle reachable on the network
- `imcpy` installed (`pip install imcpy`)

### Usage

Run from the repository root:

```bash
python3 examples/dive_follow_pipe_maneuver.py --csv examples/pipe_centerline_auv_underwater.csv --start-idx 28
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--csv` | Path to the pipe centerline CSV file (`idx, lat, lon, depth_m`) |
| `--start-idx` | Index in the CSV to start the pipe-following from |

### CSV files

| File | Description |
|------|-------------|
| `pipe_centerline_auv_underwater.csv` | Pipe waypoints at seabed depth (~20 m spacing) |
| `pipe_centerline_auv_surface.csv` | Same waypoints at surface depth (0.0 m) |
| `pipe_centerline_optimized.csv` | Full-resolution pipe centerline |

---

#### Recommendations
- The imcpy library generates stub files for the bindings, meaning that you can have autocomplete and static type checking if your IDE supports them. This can for example be [PyCharm](https://www.jetbrains.com/pycharm/) or [Jedi](https://github.com/davidhalter/jedi)-based editors.
