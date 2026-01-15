.PHONY: help waymo nuplan all dev clean

help:
	@echo "123Drive"
	@echo "==========="
	@echo "make waymo   # uv sync --extra waymo"
	@echo "make nuplan  # uv sync --extra nuplan"
	@echo "make all     # uv sync --extra all"
	@echo "make clean   # rm -rf .venv"

waymo:
	uv sync --extra waymo

nuplan:
	uv sync --extra nuplan

all:
	uv sync --extra all

dev: all

clean:
	rm -rf .venv *.egg-info
