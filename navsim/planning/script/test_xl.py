from typing import Tuple
from pathlib import Path
import logging
import os
import pdb

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


from navsim.agents.diffusiondrivev2.diffusiondrivev2_rl_config import TransfuserConfig
from navsim.agents.diffusiondrivev2.diffusiondrivev2_two_stage import V2TransfuserModel_TS

base_dir = Path.home()  # 对应 $HOME


os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
os.environ["NUPLAN_MAPS_ROOT"] = str(base_dir / "data" / "navsim_logs" / "maps")
os.environ["NAVSIM_EXP_ROOT"] = str(base_dir / "data" / "navsim_logs" / "exp")
os.environ["OPENSCENE_DATA_ROOT"] = str(base_dir / "data" / "navsim_logs")
os.environ["NAVSIM_DEVKIT_ROOT"] = str(base_dir / "project" / "DiffusionDriveV2" / "navsim")

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


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



@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
  # train_data = CacheOnlyTwoStageDataset(
  #           cache_path=cfg.cache_path,
  #           log_names=cfg.train_logs,
  #       )
  # train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True, collate_fn=collate_fn_remove_nested_none)

  # model_cfg = TransfuserConfig()
  # model = V2TransfuserModel_TS(config = cfg.agent.config)
  agent: AbstractAgent = instantiate(cfg.agent)
  model = agent._transfuser_model
  model.train()

  # 统计模型参数
  total_params = sum(p.numel() for p in model.parameters())
  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  non_trainable_params = total_params - trainable_params

  print("=" * 60)
  print("模型参数统计:")
  print("=" * 60)
  print(f"总参数量: {total_params:,} ({total_params/1e6:.2f} M)")
  print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f} M)")
  print(f"不可训练参数: {non_trainable_params:,} ({non_trainable_params/1e6:.2f} M)")
  print(f"可训练参数占比: {trainable_params/total_params*100:.2f}%")
  print("=" * 60)

  # 按模块统计参数
  print("\n按模块统计参数:")
  print("-" * 60)
  for name, module in model.named_children():
      module_params = sum(p.numel() for p in module.parameters())
      module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
      if module_params > 0:
          print(f"{name}:")
          print(f"  总参数: {module_params:,} ({module_params/1e6:.2f} M)")
          print(f"  可训练: {module_trainable:,} ({module_trainable/1e6:.2f} M)")
          print(f"  占比: {module_params/total_params*100:.2f}%")
  print("-" * 60)

  # 批量大小统计（如果需要）
  print(f"\n模型模式: {'训练模式' if model.training else '评估模式'}")
# batch = []
# for i in range(min(10, len(train_data))):
#    batch.append(train_data[i])
  # batch = []
  # for i in range(min(10, len(train_data))):
  #    batch.append(train_data[i])

  # batch = collate_fn_remove_nested_none(batch)
  # for data in batch:
  #   pdb.set_trace()
  #   feature = data[0]
  #   print(type(feature))
  #   #  print(feature)
  #   for key in feature.keys():
  #     if(feature[key] is None):
  #       print(f"\033[31m {key} is None.\033[0m")
  #       continue
  #     if isinstance(feature[key], torch.Tensor):
  #       print(f"{key} : ", feature[key].shape)
  #       continue
  #     for k in feature[key].keys():
  #       if feature[key][k] is None:
  #         print(f"\033[31m {key}_{k} is None.\033[0m")
  #       if isinstance(feature[key][k], torch.Tensor):
  #         print(f"{key}_{k} : ", feature[key][k].shape)
  
  
  # for idx, data in enumerate(train_dataloader):
  #      feature = data[0]
  #      pdm_cache = data[1]
  #      token = data[2]

  #      out = model(feature,metric_cache = data[1])

  #      pdb.set_trace()

      #  print(f"cache: {pdm_cache} , token: {token}")
      #  for key in feature.keys():
      #   if(feature[key] is None):
      #     print(f"\033[31m {key} is None.\033[0m")
      #     continue
      #   if isinstance(feature[key], torch.Tensor):
      #     print(f"{key} : ", feature[key].shape)
      #     continue
      #   for k in feature[key].keys():
      #     if feature[key][k] is None:
      #       print(f"\033[31m {key}_{k} is None.\033[0m")
      #     if isinstance(feature[key][k], torch.Tensor):
      #       print(f"{key}_{k} : ", feature[key][k].shape)



if __name__ == "__main__":

  main()

  exit()
  print(f"NAVSIM_DEVKIT_ROOT: {os.environ.get('NAVSIM_DEVKIT_ROOT')}")

  agent: AbstractAgent = instantiate(cfg.agent)
  # model = V2TransfuserModel_TS(config = model_cfg)


  feature = torch.load("./feature.pt")
  feature["trajectory"] = feature["agent"]["target"][:, :1, 4::5, :]
  print(feature["trajectory"].shape)
  target = {}
  target["trajectory"] = feature["agent"]["target"][:1, 0 , 4:41:5, :]
  
  for key in feature.keys():
    if(feature[key] is None):
      print(f"\033[31m {key} is None.\033[0m")
      continue
    if isinstance(feature[key], torch.Tensor):
      print(f"{key} : ", feature[key].shape)
      continue
    for k in feature[key].keys():
      if feature[key][k] is None:
        print(f"\033[31m {key}_{k} is None.\033[0m")
      if isinstance(feature[key][k], torch.Tensor):
        print(f"{key}_{k} : ", feature[key][k].shape)

  

  model.train()
  res = model(features =  feature, 
              targets = target,
              cal_pdm = False)
  for key in res.keys():
    if isinstance(res[key], torch.Tensor):
      print(f"{key} : ", res[key].shape)

  total_params = sum(p.numel() for p in model.parameters())
  # 格式化输出（转成百万/千，更易读）
  print(f"总参数量: {total_params:,} ({total_params/1e6:.2f} M)")

