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
