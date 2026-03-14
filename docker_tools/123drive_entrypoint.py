"""Entrypoint wrapper for the 123Drive converter Docker image."""

import os
import sys


if __name__ == "__main__":
    cmd = ["convert", "--py123d_path", "/input", "--output", "/output", *sys.argv[1:]]
    print(f"$ {' '.join(cmd)}")
    os.execvp("convert", cmd)
