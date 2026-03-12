import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.planning.training.dataset import TwoStageDataset


SPLIT = "mini"  # ["mini", "test", "trainval"]
FILTER = "all_scenes"


def _plot_agents(ax, agent, title="agents"):
    position = agent["position"].numpy() # A, T, 2
    heading = agent["heading"].numpy()
    valid = agent["valid_mask"].numpy()
    target = agent["target"].numpy()
    category = agent["category"].numpy()

    colors = {
        0: "red",    # ego
        1: "blue",   # vehicle
        2: "green",  # pedestrian
        3: "orange", # bicycle
    }

    for i in range(position.shape[0]):
        if not valid[i].any():
            continue
        color = colors.get(int(category[i]), "gray")
        pts = position[i][valid[i]]
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=1.5)
        ax.plot(pts[-1, 0], pts[-1, 1], "o", color=color, markersize=4)

        if target.shape[1] > 0:
            ax.plot(target[i, :, 0], target[i, :, 1], "--", color=color, alpha=0.6, linewidth=1.0)

        if pts.shape[0] > 0:
            ax.arrow(
                pts[-1, 0],
                pts[-1, 1],
                0.8 * np.cos(heading[i, valid[i]].astype(float)[-1]),
                0.8 * np.sin(heading[i, valid[i]].astype(float)[-1]),
                head_width=0.4,
                head_length=0.6,
                color=color,
                alpha=0.7,
                length_includes_head=True,
            )

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.4)


def _plot_map(ax, map_data, reference_line=None, static_objects=None, title="map"):
    point_position = map_data["point_position"].numpy() # M, 3, P, 2
    valid_mask = map_data["valid_mask"].numpy()
    polygon_tl_status = map_data.get("polygon_tl_status", None)
    if polygon_tl_status is not None:
        polygon_tl_status = polygon_tl_status.numpy()

    tl_colors = {
        1: "green",
        2: "yellow",
        3: "red",
    }

    for i in range(point_position.shape[0]):
        if not valid_mask[i].any():
            continue
        tl_color = None
        if polygon_tl_status is not None:
            tl_color = tl_colors.get(int(polygon_tl_status[i]), None)
        for side_idx, color in zip([0, 1, 2], ["black", "gray", "gray"]):
            pts = point_position[i, side_idx]
            use_color = tl_color if (side_idx == 0 and tl_color is not None) else color
            ax.plot(pts[:, 0], pts[:, 1], "-", color=use_color, linewidth=0.6, alpha=0.9)

    if reference_line is not None:
        ref_pos = reference_line["position"].numpy()
        ref_mask = reference_line["valid_mask"].numpy()
        for i in range(ref_pos.shape[0]):
            if not ref_mask[i].any():
                continue
            pts = ref_pos[i][ref_mask[i]]
            ax.plot(pts[:, 0], pts[:, 1], "-", color="purple", linewidth=1.2, alpha=0.9)

    if static_objects is not None:
        _plot_static_objects(ax, static_objects)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.4)


def _plot_static_objects(ax, static_objects):
    positions = static_objects["position"].numpy()
    headings = static_objects["heading"].numpy()
    shapes = static_objects["shape"].numpy()
    valid_mask = static_objects["valid_mask"].numpy()

    for i in range(positions.shape[0]):
        if not valid_mask[i]:
            continue
        x, y = positions[i]
        width, length = shapes[i]
        heading = headings[i]
        corners = _oriented_box_corners(x, y, width, length, heading)
        xs, ys = zip(*(corners + [corners[0]]))
        ax.plot(xs, ys, "-", color="brown", linewidth=1.0, alpha=0.9)


def _oriented_box_corners(x, y, width, length, heading):
    half_w = width / 2.0
    half_l = length / 2.0
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    local = [
        (half_l, half_w),
        (half_l, -half_w),
        (-half_l, -half_w),
        (-half_l, half_w),
    ]
    corners = []
    for lx, ly in local:
        gx = x + lx * cos_h - ly * sin_h
        gy = y + lx * sin_h + ly * cos_h
        corners.append((gx, gy))
    return corners


def _build_legends(fig, axes):
    agent_legend = [
        Line2D([0], [0], color="red", lw=2, label="Ego"),
        Line2D([0], [0], color="blue", lw=2, label="Vehicle"),
        Line2D([0], [0], color="green", lw=2, label="Pedestrian"),
        Line2D([0], [0], color="orange", lw=2, label="Bicycle"),
    ]

    static_legend = [
        Line2D([0], [0], color="brown", lw=2, label="Static Object"),
    ]

    tl_legend = [
        Line2D([0], [0], color="green", lw=2, label="Traffic Light: Green"),
        Line2D([0], [0], color="yellow", lw=2, label="Traffic Light: Yellow"),
        Line2D([0], [0], color="red", lw=2, label="Traffic Light: Red"),
        Line2D([0], [0], color="black", lw=2, label="Lane Center/Boundary"),
        Line2D([0], [0], color="purple", lw=2, label="Reference Line"),
    ]

    axes[0].legend(handles=agent_legend, loc="upper right", framealpha=0.8)
    axes[1].legend(handles=tl_legend + static_legend, loc="upper right", framealpha=0.8)


def main():
    hydra.initialize(config_path="../navsim/planning/script/config/common/train_test_split/scene_filter")
    cfg = hydra.compose(config_name=FILTER)
    scene_filter: SceneFilter = instantiate(cfg)
    openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))

    scene_loader = SceneLoader(
        openscene_data_root / f"navsim_logs/{SPLIT}",
        openscene_data_root / f"sensor_blobs/{SPLIT}",
        scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )

    dataset = TwoStageDataset(scene_loader=scene_loader, build_reference_line=True)

    sample = dataset[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    _plot_agents(axes[0], sample["agent"], title="Agent (history + future)")
    _plot_map(
        axes[1],
        sample["map"],
        reference_line=sample.get("reference_line", None),
        static_objects=sample.get("static_objects", None),
        title="Map / Reference Line / Static Objects",
    )
    _build_legends(fig, axes)
    plt.tight_layout()
    # plt.show()
    plt.savefig("./two_stage_vis.png")


if __name__ == "__main__":
    # export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
    # export NUPLAN_MAPS_ROOT="$HOME/data/navsim_logs/maps"
    # export NAVSIM_EXP_ROOT="$HOME/data/navsim_logs/exp"
    # export NAVSIM_DEVKIT_ROOT="$HOME/project/DiffusionDriveV2/navsim"
    # export OPENSCENE_DATA_ROOT="$HOME/ndata/navsim_logs"

    import os
    from pathlib import Path

    base_dir = Path.home()  # 对应 $HOME


    os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
    os.environ["NUPLAN_MAPS_ROOT"] = str(base_dir / "data" / "navsim_logs" / "maps")
    os.environ["NAVSIM_EXP_ROOT"] = str(base_dir / "data" / "navsim_logs" / "exp")
    os.environ["OPENSCENE_DATA_ROOT"] = str(base_dir / "data" / "navsim_logs")
    os.environ["NAVSIM_DEVKIT_ROOT"] = str(base_dir / "project" / "DiffusionDriveV2" / "navsim")


    print(f"NAVSIM_DEVKIT_ROOT: {os.environ.get('NAVSIM_DEVKIT_ROOT')}")
    main()
