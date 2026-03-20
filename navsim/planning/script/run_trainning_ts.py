from typing import Tuple
from pathlib import Path
import logging

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset_two_stage import CacheOnlyTwoStageDataset, TwoStageDataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

def _log_torch_device_info() -> None:
    # Keep this lightweight and robust: CUDA/NVML visibility differs across environments (docker, slurm, etc).
    logger.info("Torch version: %s", torch.__version__)
    logger.info("CUDA available (torch.cuda.is_available): %s", torch.cuda.is_available())
    try:
        logger.info("CUDA device_count: %s", torch.cuda.device_count())
        if torch.cuda.device_count() > 0:
            logger.info("CUDA current_device: %s", torch.cuda.current_device())
            logger.info("CUDA device_name[0]: %s", torch.cuda.get_device_name(0))
    except Exception as e:
        logger.warning("Failed to query torch.cuda device details: %s", e)


def remove_none_recursive(data):
    if isinstance(data, dict):
        filtered_dict = {}
        for k, v in data.items():
            processed_v = remove_none_recursive(v)
            if processed_v is not None:
                filtered_dict[k] = processed_v
        return filtered_dict
    elif isinstance(data, list):
        return [remove_none_recursive(item) for item in data]
    else:
        return data


def collate_fn_remove_nested_none(batch):
   
    # processed_batch = [remove_none_recursive(sample) for sample in batch]
    processed_batch = [(remove_none_recursive(x), y, z) for (x, y, z) in batch]
    # return processed_batch
    return torch.utils.data.default_collate(processed_batch)

def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[TwoStageDataset, TwoStageDataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = TwoStageDataset(
        scene_loader=train_scene_loader,
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = TwoStageDataset(
        scene_loader=val_scene_loader,
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    torch.set_float32_matmul_precision('medium') # | 'high')
    
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")
    _log_torch_device_info()

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyTwoStageDataset(
            cache_path=cfg.cache_path,
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyTwoStageDataset(
            cache_path=cfg.cache_path,
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.train_dataloader.params, shuffle=True, collate_fn=collate_fn_remove_nested_none)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.val_dataloader.params, shuffle=False, collate_fn=collate_fn_remove_nested_none)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
