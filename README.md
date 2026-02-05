# 123Drive

Converts [py123d](https://github.com/autonomousvision/py123d) data to [PufferDrive](https://github.com/Emerge-Lab/PufferDrive) binary format.

## Usage

```bash
# Basic conversion
uv run convert --dataset_path /path/to/data --output_dir ./output

# With processors
uv run convert --dataset_path /path/to/data --output_dir ./output --interpolate --traffic_lights --validate

# Parallel processing
uv run convert --dataset_path /path/to/data --output_dir ./output --num_workers 8 --batch_size 10
```

### Options

| Flag | Description |
|------|-------------|
| `--interpolate` | Densify road geometry polylines |
| `--traffic_lights` | Generate synthetic traffic light sequences |
| `--validate` | Validate output before binary encoding |
| `--max_scenarios N` | Limit number of scenarios to process |
| `--map_only` | Load map-only scenarios (no logs) |


## License

See LICENSE file for details.
