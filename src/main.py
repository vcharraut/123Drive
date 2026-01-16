import logging
import os
import warnings

import hydra
from omegaconf import DictConfig, OmegaConf

from src import logger_utils
from src.core import json_to_bin, pipeline


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

warnings.filterwarnings("ignore", category=UserWarning, module=".*tensorflow.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*tensorboard.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*tensorflow.*")

logging.getLogger("tensorflow").setLevel(logging.FATAL)
logging.getLogger("absl").setLevel(logging.FATAL)

logger = logger_utils.get_logger(__name__)


def _validate_cfg(cfg: DictConfig):
    if cfg.command == "json_to_bin":
        if not cfg.json_to_bin.input_dir:
            raise ValueError("json_to_bin.input_dir is required")
        if not cfg.json_to_bin.output_dir:
            raise ValueError("json_to_bin.output_dir is required")
        return

    if not cfg.dataset_path:
        raise ValueError("dataset_path is required")


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    logger_utils.setup_logger()
    _validate_cfg(cfg)

    logger.info("123Drive")
    logger.info(OmegaConf.to_yaml(cfg))

    if cfg.command == "convert":
        pipeline.run(cfg)
    elif cfg.command == "json_to_bin":
        json_to_bin.run(cfg)
    else:
        raise ValueError(f"Unknown command: {cfg.command} (supported: convert|json_to_bin)")
    return 0


def cli_entry_point():
    return main()


if __name__ == "__main__":
    raise SystemExit(cli_entry_point())
