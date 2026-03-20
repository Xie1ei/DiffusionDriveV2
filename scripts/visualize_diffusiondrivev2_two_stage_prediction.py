import argparse
import os
from pathlib import Path
import logging

base_dir = Path.home()  # 对应 $HOME
os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
os.environ["NUPLAN_MAPS_ROOT"] = str(base_dir / "data" / "navsim_logs" / "maps")
os.environ["NAVSIM_EXP_ROOT"] = str(base_dir / "data" / "navsim_logs" / "exp")
os.environ["OPENSCENE_DATA_ROOT"] = str(base_dir / "data" / "navsim_logs")
os.environ["NAVSIM_DEVKIT_ROOT"] = str(base_dir / "project" / "DiffusionDriveV2" / "navsim")

import hydra
from hydra.utils import instantiate
import matplotlib.pyplot as plt

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.planning.training.dataset_two_stage import TwoStageDataset
from navsim.visualization.diffusiondrivev2_two_stage import (
    visualize_diffusiondrivev2_two_stage_prediction,
)




print(f"NAVSIM_DEVKIT_ROOT: {os.environ.get('NAVSIM_DEVKIT_ROOT')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize DiffusionDriveV2 two-stage predictions.")
    parser.add_argument("--token", required=False, help="Scene token to visualize. If omitted, use the first token.")
    parser.add_argument("--split", default="mini", help="Dataset split, e.g. mini/test/trainval.")
    parser.add_argument("--filter", default="all_scenes", help="Scene filter config name.")
    parser.add_argument("--checkpoint-path", default=None, help="Optional model checkpoint path.")
    parser.add_argument("--output", default=None, help="Optional path to save the figure.")
    parser.add_argument("--top-k-anchors", type=int, default=3, help="How many anchors to plot.")
    return parser.parse_args()


def main() -> None:
    # logging.basicConfig(
    # level=logging.DEBUG,
    # format="%(asctime)s - %(levelname)s - %(message)s",
    # handlers=[
    #         logging.FileHandler("vis_debug.log"),  # DEBUG日志保存到文件
    #         logging.StreamHandler()  # 控制台也输出（可选，生产环境可注释）
    #     ]
    # )
    
    args = parse_args()

    hydra.initialize(
        config_path="../navsim/planning/script/config/common/train_test_split/scene_filter",
        version_base=None,
    )
    cfg = hydra.compose(config_name=args.filter)
    scene_filter: SceneFilter = instantiate(cfg)

    openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))
    scene_loader = SceneLoader(
        openscene_data_root / f"navsim_logs/{args.split}",
        openscene_data_root / f"sensor_blobs/{args.split}",
        scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    dataset = TwoStageDataset(scene_loader=scene_loader)

    token = args.token or scene_loader.tokens[2]
    for i in range(min(100, len(scene_loader.tokens))):
        token = scene_loader.tokens[i]
        result = visualize_diffusiondrivev2_two_stage_prediction(
            token=token,
            scene_loader=scene_loader,
            dataset=dataset,
            checkpoint_path=args.checkpoint_path,
            top_k_anchors=args.top_k_anchors,
        )

        if args.output:
            output_path = Path(args.output)
            new_filename = f"{output_path.stem}_{i}{output_path.suffix}"
            output_path = output_path.parent / new_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.fig.savefig(output_path, dpi=200, bbox_inches="tight")
        else:
            plt.show()


if __name__ == "__main__":
    main()
