from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import pickle
import gzip
import os

import torch
from tqdm import tqdm
import numpy as np
from shapely.geometry import LineString, Polygon, Point
import shapely

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import SemanticMapLayer, PolygonMapObject
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType

from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.common.dataloader import SceneLoader, MetricCacheLoader
from navsim.common.dataclasses import AgentInput, Scene, NAVSIM_INTERVAL_LENGTH
from navsim.common.enums import BoundingBoxIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import normalize_angle

logger = logging.getLogger(__name__)


def load_feature_target_from_pickle(path: Path) -> Dict[str, torch.Tensor]:
    """Helper function to load pickled feature/target from path."""
    with gzip.open(path, "rb") as f:
        data_dict: Dict[str, torch.Tensor] = pickle.load(f)
    return data_dict


def dump_feature_target_to_pickle(path: Path, data_dict: Dict[str, torch.Tensor]) -> None:
    """Helper function to save feature/target to pickle."""
    # Use compresslevel = 1 to compress the size but also has fast write and read.
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)



class TwoStageDataset(torch.utils.data.Dataset):
    """
    Dataset for two-stage end-to-end model inputs.
    Builds samples following the structure in navsim/planning/training/data_struct.md.
    """

    _CACHE_FILE_NAME = "two_stage_data.gz"

    def __init__(
        self,
        scene_loader: SceneLoader,
        cache_path: Optional[str] = None,
        force_cache_computation: bool = False,
        max_agents: int = 20,
        max_static_objects: int = 10,
        max_map_elements: int = 256,
        map_point_num: int = 20,
        map_radius: float = 50.0,
        build_reference_line: bool = False,
    ):
        super().__init__()
        self._scene_loader = scene_loader
        self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None
        self._force_cache_computation = force_cache_computation

        self._max_agents = max_agents
        self._max_static_objects = max_static_objects
        self._max_map_elements = max_map_elements
        self._map_point_num = map_point_num
        self._map_radius = map_radius
        self._build_reference_line = build_reference_line

        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(self._cache_path)
        if self._cache_path is not None:
            self.cache_dataset()

    @staticmethod
    def _load_valid_caches(cache_path: Optional[Path]) -> Dict[str, Path]:
        valid_cache_paths: Dict[str, Path] = {}
        if (cache_path is not None) and cache_path.is_dir():
            for log_path in cache_path.iterdir():
                for token_path in log_path.iterdir():
                    data_dict_path = token_path / TwoStageDataset._CACHE_FILE_NAME
                    if data_dict_path.is_file():
                        valid_cache_paths[token_path.name] = token_path
        return valid_cache_paths

    def __len__(self) -> int:
        return len(self._scene_loader)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        token = self._scene_loader.tokens[idx]
        if self._cache_path is not None:
            assert (
                token in self._valid_cache_paths
            ), f"The token {token} has not been cached yet, please call cache_dataset first!"
            return self._load_scene_with_token(token)

        scene = self._scene_loader.get_scene_from_token(token)
        return self._build_sample(scene)

    def cache_dataset(self) -> None:
        assert self._cache_path is not None, "Dataset did not receive a cache path!"
        os.makedirs(self._cache_path, exist_ok=True)

        if self._force_cache_computation:
            tokens_to_cache = self._scene_loader.tokens
        else:
            tokens_to_cache = set(self._scene_loader.tokens) - set(self._valid_cache_paths.keys())
            tokens_to_cache = list(tokens_to_cache)
            logger.info(
                f"""
                Starting caching of {len(tokens_to_cache)} tokens.
                Note: Caching tokens within the training loader is slow. Only use it with a small number of tokens.
                You can cache large numbers of tokens using the `run_dataset_caching.py` python script.
                """
            )

        for token in tqdm(tokens_to_cache, desc="Caching Two-Stage Dataset"):
            self._cache_scene_with_token(token)

    def _cache_scene_with_token(self, token: str) -> None:
        scene = self._scene_loader.get_scene_from_token(token)
        data_dict = self._build_sample(scene)

        metadata = scene.scene_metadata
        token_path = self._cache_path / metadata.log_name / metadata.initial_token
        os.makedirs(token_path, exist_ok=True)

        data_dict_path = token_path / self._CACHE_FILE_NAME
        dump_feature_target_to_pickle(data_dict_path, data_dict)
        self._valid_cache_paths[token] = token_path

    def _load_scene_with_token(self, token: str) -> Dict[str, Any]:
        token_path = self._valid_cache_paths[token]
        data_dict_path = token_path / self._CACHE_FILE_NAME
        return load_feature_target_from_pickle(data_dict_path)

    def _build_sample(self, scene: Scene) -> Dict[str, Any]:
        frame_idx = scene.scene_metadata.num_history_frames - 1
        current_frame = scene.frames[frame_idx]
        current_ego_pose = current_frame.ego_status.ego_pose

        agent_input: AgentInput = scene.get_agent_input()

        data = {
            "agent": self._build_agent_data(scene, current_ego_pose),
            "map": self._build_map_data(
                scene, current_ego_pose, current_frame.roadblock_ids, current_frame.traffic_lights
            ),
            "static_objects": self._build_static_objects(current_frame.annotations),
            "reference_line": None,
            "current_state": self._build_current_state(agent_input),
            "causal": {},
            "cost_maps": {},
        }

        if self._build_reference_line:
            data["reference_line"] = self._build_reference_line_feature(
                scene, current_ego_pose, current_frame.roadblock_ids
            )

        return data

    def _build_current_state(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        ego_status = agent_input.ego_statuses[-1]
        return {
            "position": torch.tensor(ego_status.ego_pose[:2], dtype=torch.float32),
            "heading": torch.tensor(ego_status.ego_pose[2], dtype=torch.float32),
            "velocity": torch.tensor(ego_status.ego_velocity, dtype=torch.float32),
            "acceleration": torch.tensor(ego_status.ego_acceleration, dtype=torch.float32),
        }

    def _build_agent_data(self, scene: Scene, current_ego_pose: np.ndarray) -> Dict[str, torch.Tensor]:
        num_history = scene.scene_metadata.num_history_frames
        num_future = scene.scene_metadata.num_future_frames

        current_frame_idx = num_history - 1
        current_annotations = scene.frames[current_frame_idx].annotations

        # select agents from current frame (exclude ego, keep vehicle/pedestrian/bicycle)
        candidate = []
        for box, name, track_token in zip(
            current_annotations.boxes, current_annotations.names, current_annotations.track_tokens
        ):
            if name not in ("vehicle", "pedestrian", "bicycle"):
                continue
            x, y = box[BoundingBoxIndex.X], box[BoundingBoxIndex.Y]
            candidate.append((float(np.hypot(x, y)), track_token, name))

        candidate.sort(key=lambda x: x[0])
        selected = candidate[: max(self._max_agents - 1, 0)]

        track_tokens = [t[1] for t in selected]
        categories = [t[2] for t in selected]

        A = self._max_agents
        T = num_history
        Tf = num_future

        position = np.zeros((A, T, 2), dtype=np.float32)
        heading = np.zeros((A, T), dtype=np.float32)
        velocity = np.zeros((A, T, 2), dtype=np.float32)
        shape = np.zeros((A, T, 2), dtype=np.float32)
        valid_mask = np.zeros((A, T), dtype=bool)

        target = np.zeros((A, Tf, 3), dtype=np.float32)

        category = np.zeros((A,), dtype=np.int64)
        category[0] = 0  # ego

        for i, name in enumerate(categories, start=1):
            if name == "vehicle":
                category[i] = 1
            elif name == "pedestrian":
                category[i] = 2
            elif name == "bicycle":
                category[i] = 3
            else:
                category[i] = 0

        current_pose = current_ego_pose

        # ego history (positions already in current ego frame)
        ego_history = scene.get_history_trajectory().poses
        position[0] = ego_history[:, :2]
        heading[0] = ego_history[:, 2]
        for t in range(T):
            status = scene.frames[t].ego_status
            gvx, gvy = self._rotate_vector(status.ego_velocity[0], status.ego_velocity[1], status.ego_pose[2])
            rvx, rvy = self._rotate_vector(gvx, gvy, -current_pose[2])
            velocity[0, t] = [rvx, rvy]
            shape[0, t] = np.array([get_pacifica_parameters().width, get_pacifica_parameters().length], dtype=np.float32)
            valid_mask[0, t] = True

        # ego future target
        ego_future = scene.get_future_trajectory(num_trajectory_frames=Tf).poses
        target[0, : len(ego_future)] = ego_future

        # build per-frame lookup for selected agents
        for t in range(len(scene.frames)):
            frame = scene.frames[t]
            frame_pose = frame.ego_status.ego_pose
            token_to_idx = {tok: i + 1 for i, tok in enumerate(track_tokens)}
            for box, name, vel, track_token in zip(
                frame.annotations.boxes,
                frame.annotations.names,
                frame.annotations.velocity_3d,
                frame.annotations.track_tokens,
            ):
                if track_token not in token_to_idx:
                    continue

                idx = token_to_idx[track_token]

                local_x, local_y, local_heading = (
                    box[BoundingBoxIndex.X],
                    box[BoundingBoxIndex.Y],
                    box[BoundingBoxIndex.HEADING],
                )
                global_x, global_y, global_heading = self._local_to_global(
                    local_x, local_y, local_heading, frame_pose
                )
                rel_x, rel_y, rel_heading = self._global_to_local(
                    global_x, global_y, global_heading, current_pose
                )

                # velocity: local -> global -> current local
                vx, vy = float(vel[0]), float(vel[1])
                gvx, gvy = self._rotate_vector(vx, vy, frame_pose[2])
                rvx, rvy = self._rotate_vector(gvx, gvy, -current_pose[2])

                width = box[BoundingBoxIndex.WIDTH]
                length = box[BoundingBoxIndex.LENGTH]

                if t < T:
                    position[idx, t] = [rel_x, rel_y]
                    heading[idx, t] = rel_heading
                    velocity[idx, t] = [rvx, rvy]
                    shape[idx, t] = [width, length]
                    valid_mask[idx, t] = True
                else:
                    f_idx = t - T
                    if f_idx < Tf:
                        target[idx, f_idx] = [rel_x, rel_y, rel_heading]

        return {
            "position": torch.tensor(position, dtype=torch.float32),
            "heading": torch.tensor(heading, dtype=torch.float32),
            "velocity": torch.tensor(velocity, dtype=torch.float32),
            "shape": torch.tensor(shape, dtype=torch.float32),
            "category": torch.tensor(category, dtype=torch.int64),
            "valid_mask": torch.tensor(valid_mask, dtype=torch.bool),
            "target": torch.tensor(target, dtype=torch.float32),
        }

    def _build_static_objects(self, annotations) -> Dict[str, torch.Tensor]:
        static_types = {
            "czone_sign": 0,
            "barrier": 1,
            "traffic_cone": 2,
            "generic_object": 3,
        }

        positions = np.zeros((self._max_static_objects, 2), dtype=np.float32)
        headings = np.zeros((self._max_static_objects,), dtype=np.float32)
        shapes = np.zeros((self._max_static_objects, 2), dtype=np.float32)
        categories = np.zeros((self._max_static_objects,), dtype=np.int64)
        valid_mask = np.zeros((self._max_static_objects,), dtype=bool)

        count = 0
        for box, name in zip(annotations.boxes, annotations.names):
            if name not in static_types:
                continue
            if count >= self._max_static_objects:
                break
            positions[count] = [box[BoundingBoxIndex.X], box[BoundingBoxIndex.Y]]
            headings[count] = box[BoundingBoxIndex.HEADING]
            shapes[count] = [box[BoundingBoxIndex.WIDTH], box[BoundingBoxIndex.LENGTH]]
            categories[count] = static_types[name]
            valid_mask[count] = True
            count += 1

        return {
            "position": torch.tensor(positions, dtype=torch.float32),
            "heading": torch.tensor(headings, dtype=torch.float32),
            "shape": torch.tensor(shapes, dtype=torch.float32),
            "category": torch.tensor(categories, dtype=torch.int64),
            "valid_mask": torch.tensor(valid_mask, dtype=torch.bool),
        }

    def _build_map_data(
        self,
        scene: Scene,
        current_ego_pose: np.ndarray,
        route_roadblock_ids: List[str],
        traffic_lights: List[Tuple[str, bool]],
    ) -> Dict[str, torch.Tensor]:
        map_api = scene.map_api
        ego_pose = StateSE2(*current_ego_pose)

        layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR, SemanticMapLayer.CROSSWALK]
        map_object_dict = map_api.get_proximal_map_objects(point=ego_pose.point, radius=self._map_radius, layers=layers)

        objects = []
        for layer in layers:
            for obj in map_object_dict[layer]:
                objects.append((layer, obj))

        # sort by distance to ego
        obj_with_dist = []
        for layer, obj in objects:
            center = self._get_object_center_local(obj, ego_pose)
            obj_with_dist.append((float(np.hypot(center[0], center[1])), layer, obj))
        obj_with_dist.sort(key=lambda x: x[0])

        obj_with_dist = obj_with_dist[: self._max_map_elements]
        M = self._max_map_elements
        P = self._map_point_num

        point_position = np.zeros((M, 3, P, 2), dtype=np.float32)
        point_vector = np.zeros((M, 3, P, 2), dtype=np.float32)
        point_orientation = np.zeros((M, 3, P), dtype=np.float32)
        point_side = np.zeros((M, 3), dtype=np.int64)

        polygon_center = np.zeros((M, 3), dtype=np.float32)
        polygon_position = np.zeros((M, 2), dtype=np.float32)
        polygon_orientation = np.zeros((M,), dtype=np.float32)
        polygon_type = np.zeros((M,), dtype=np.int64)
        polygon_on_route = np.zeros((M,), dtype=bool)
        polygon_tl_status = np.zeros((M,), dtype=np.int64)
        polygon_has_speed_limit = np.zeros((M,), dtype=bool)
        polygon_speed_limit = np.zeros((M,), dtype=np.float32)
        polygon_road_block_id = np.full((M,), -1, dtype=np.int64)

        # Point-level validity mask for centerline points (consistent with PlutoFeature.normalize).
        valid_mask = np.zeros((M, P), dtype=bool)

        roadblock_id_map: Dict[str, int] = {}
        next_rb_id = 0

        tls = self._build_traffic_light_status(traffic_lights)

        for i, (_, layer, obj) in enumerate(obj_with_dist):
            center_line, left_line, right_line = self._get_map_lines(obj, ego_pose)

            center_points = None
            for side_idx, line in enumerate([center_line, left_line, right_line]):
                points = self._sample_linestring(line, P)
                vectors = self._points_to_vectors(points)
                orientations = self._vectors_to_orientations(vectors)

                point_position[i, side_idx] = points
                point_vector[i, side_idx] = vectors
                point_orientation[i, side_idx] = orientations
                point_side[i, side_idx] = side_idx
                if side_idx == 0:
                    center_points = points

            if center_points is not None and center_line.length > 0:
                within_x = (center_points[:, 0] < self._map_radius) & (
                    center_points[:, 0] > -self._map_radius
                )
                within_y = (center_points[:, 1] < self._map_radius) & (
                    center_points[:, 1] > -self._map_radius
                )
                valid_mask[i] = within_x & within_y

            center = self._get_object_center_local(obj, ego_pose)
            polygon_center[i] = center
            polygon_position[i] = point_position[i, 0, 0]
            polygon_orientation[i] = point_orientation[i, 0, 0]

            if layer == SemanticMapLayer.LANE:
                polygon_type[i] = 0
            elif layer == SemanticMapLayer.LANE_CONNECTOR:
                polygon_type[i] = 1
            else:
                polygon_type[i] = 2

            # route info
            rb_id = self._get_roadblock_id(obj)
            if rb_id is not None:
                polygon_on_route[i] = rb_id in route_roadblock_ids
                if rb_id not in roadblock_id_map:
                    roadblock_id_map[rb_id] = next_rb_id
                    next_rb_id += 1
                polygon_road_block_id[i] = roadblock_id_map[rb_id]

            # traffic light status (unknown if no reliable mapping)
            if layer == SemanticMapLayer.LANE_CONNECTOR:
                object_id = self._get_map_object_id(obj)
                if object_id is not None and object_id in tls:
                    polygon_tl_status[i] = tls[object_id]
                else:
                    polygon_tl_status[i] = int(TrafficLightStatusType.UNKNOWN)
            else:
                polygon_tl_status[i] = int(TrafficLightStatusType.UNKNOWN)

            # speed limit
            speed = getattr(obj, "speed_limit_mps", None)
            if speed is not None:
                polygon_has_speed_limit[i] = True
                polygon_speed_limit[i] = float(speed)

        return {
            "point_position": torch.tensor(point_position, dtype=torch.float32),
            "point_vector": torch.tensor(point_vector, dtype=torch.float32),
            "point_orientation": torch.tensor(point_orientation, dtype=torch.float32),
            "point_side": torch.tensor(point_side, dtype=torch.int64),
            "polygon_center": torch.tensor(polygon_center, dtype=torch.float32),
            "polygon_position": torch.tensor(polygon_position, dtype=torch.float32),
            "polygon_orientation": torch.tensor(polygon_orientation, dtype=torch.float32),
            "polygon_type": torch.tensor(polygon_type, dtype=torch.int64),
            "polygon_on_route": torch.tensor(polygon_on_route, dtype=torch.bool),
            "polygon_tl_status": torch.tensor(polygon_tl_status, dtype=torch.int64),
            "polygon_has_speed_limit": torch.tensor(polygon_has_speed_limit, dtype=torch.bool),
            "polygon_speed_limit": torch.tensor(polygon_speed_limit, dtype=torch.float32),
            "polygon_road_block_id": torch.tensor(polygon_road_block_id, dtype=torch.int64),
            "valid_mask": torch.tensor(valid_mask, dtype=torch.bool),
        }

    @staticmethod
    def _local_to_global(x: float, y: float, heading: float, ego_pose: np.ndarray) -> Tuple[float, float, float]:
        cos_h = np.cos(ego_pose[2])
        sin_h = np.sin(ego_pose[2])
        gx = x * cos_h - y * sin_h + ego_pose[0]
        gy = x * sin_h + y * cos_h + ego_pose[1]
        g_heading = normalize_angle(heading + ego_pose[2])
        return gx, gy, g_heading

    @staticmethod
    def _global_to_local(x: float, y: float, heading: float, ego_pose: np.ndarray) -> Tuple[float, float, float]:
        dx = x - ego_pose[0]
        dy = y - ego_pose[1]
        cos_h = np.cos(ego_pose[2])
        sin_h = np.sin(ego_pose[2])
        lx = dx * cos_h + dy * sin_h
        ly = -dx * sin_h + dy * cos_h
        l_heading = normalize_angle(heading - ego_pose[2])
        return lx, ly, l_heading

    @staticmethod
    def _rotate_vector(x: float, y: float, angle: float) -> Tuple[float, float]:
        cos_h = np.cos(angle)
        sin_h = np.sin(angle)
        rx = x * cos_h - y * sin_h
        ry = x * sin_h + y * cos_h
        return rx, ry

    def _get_map_lines(
        self, obj: Any, ego_pose: StateSE2
    ) -> Tuple[LineString, LineString, LineString]:
        if isinstance(obj, LaneGraphEdgeMapObject):
            center = self._states_to_linestring(obj.baseline_path.discrete_path, ego_pose)
            left = self._states_to_linestring(obj.left_boundary.discrete_path, ego_pose)
            right = self._states_to_linestring(obj.right_boundary.discrete_path, ego_pose)
            return center, left, right

        if hasattr(obj, "polygon"):
            polygon = self._geometry_local_coords(obj.polygon, ego_pose)
            edges = self._get_crosswalk_edges(polygon, self._map_point_num + 1)
            center = LineString(edges[0])
            left = LineString(edges[1])
            right = LineString(edges[2])
            return center, left, right

        if hasattr(obj, "baseline_path"):
            center = self._geometry_local_coords(obj.baseline_path.linestring, ego_pose)
            return center, center, center

        center = LineString([])
        return center, center, center

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        from shapely import affinity

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])
        return rotated_geometry

    def _sample_linestring(self, line: LineString, num_points: int) -> np.ndarray:
        if line is None or line.length == 0:
            return np.zeros((num_points, 2), dtype=np.float32)

        distances = np.linspace(0.0, line.length, num_points)
        points = [line.interpolate(d) for d in distances]
        coords = np.array([[p.x, p.y] for p in points], dtype=np.float32)
        return coords

    def _states_to_linestring(self, states: List[StateSE2], ego_pose: StateSE2) -> LineString:
        points = []
        for state in states:
            ego_arr = np.array([ego_pose.x, ego_pose.y, ego_pose.heading], dtype=np.float32)
            lx, ly, _ = self._global_to_local(state.x, state.y, state.heading, ego_arr)
            points.append((lx, ly))
        if not points:
            return LineString([])
        return LineString(points)

    @staticmethod
    def _get_crosswalk_edges(polygon: Polygon, sample_points: int) -> np.ndarray:
        bbox = shapely.minimum_rotated_rectangle(polygon)
        coords = np.stack(bbox.exterior.coords.xy, axis=-1)
        edge1 = coords[[3, 0]]  # right boundary
        edge2 = coords[[2, 1]]  # left boundary

        edges = np.stack([(edge1 + edge2) * 0.5, edge2, edge1], axis=0)
        vector = edges[:, 1] - edges[:, 0]
        steps = np.linspace(0, 1, sample_points, endpoint=True)[None, :]
        points = edges[:, 0][:, None, :] + vector[:, None, :] * steps[:, :, None]
        return points

    @staticmethod
    def _points_to_vectors(points: np.ndarray) -> np.ndarray:
        vectors = np.zeros_like(points)
        if len(points) > 1:
            vectors[:-1] = points[1:] - points[:-1]
            vectors[-1] = vectors[-2]
        return vectors

    @staticmethod
    def _vectors_to_orientations(vectors: np.ndarray) -> np.ndarray:
        return np.arctan2(vectors[:, 1], vectors[:, 0]).astype(np.float32)

    def _get_object_center_local(self, obj: Any, ego_pose: StateSE2) -> np.ndarray:
        if hasattr(obj, "polygon"):
            polygon: Polygon = self._geometry_local_coords(obj.polygon, ego_pose)
            center = polygon.centroid
            center_xy = np.array([center.x, center.y], dtype=np.float32)
        else:
            center_line, _, _ = self._get_map_lines(obj, ego_pose)
            if center_line.length == 0:
                center_xy = np.zeros(2, dtype=np.float32)
            else:
                mid = center_line.interpolate(center_line.length * 0.5)
                center_xy = np.array([mid.x, mid.y], dtype=np.float32)

        center_line, _, _ = self._get_map_lines(obj, ego_pose)
        if center_line.length == 0:
            heading = 0.0
        else:
            points = self._sample_linestring(center_line, 2)
            vec = points[1] - points[0]
            heading = float(np.arctan2(vec[1], vec[0]))

        return np.array([center_xy[0], center_xy[1], heading], dtype=np.float32)

    @staticmethod
    def _get_roadblock_id(obj: Any) -> Optional[str]:
        if hasattr(obj, "get_roadblock_id"):
            try:
                return obj.get_roadblock_id()
            except Exception:
                return None
        return None

    @staticmethod
    def _get_map_object_id(obj: Any) -> Optional[int]:
        if hasattr(obj, "id"):
            try:
                return int(obj.id)
            except Exception:
                return None
        return None

    @staticmethod
    def _build_traffic_light_status(traffic_lights: List[Tuple[str, bool]]) -> Dict[int, int]:
        tls: Dict[int, int] = {}
        for tl in traffic_lights:
            if isinstance(tl, tuple) and len(tl) == 2:
                tl_id, tl_state = tl
                try:
                    tl_id_int = int(tl_id)
                except Exception:
                    continue
                if isinstance(tl_state, bool):
                    tls[tl_id_int] = int(
                        TrafficLightStatusType.GREEN if tl_state else TrafficLightStatusType.RED
                    )
                elif isinstance(tl_state, (int, np.integer)):
                    tls[tl_id_int] = int(tl_state)
                else:
                    tls[tl_id_int] = int(TrafficLightStatusType.UNKNOWN)
        return tls

    def _build_reference_line_feature(
        self, scene: Scene, current_ego_pose: np.ndarray, route_roadblock_ids: List[str]
    ) -> Dict[str, torch.Tensor]:
        map_api = scene.map_api
        ego_pose = StateSE2(*current_ego_pose)

        reference_lines = []
        for rb_id in route_roadblock_ids:
            roadblock = map_api.get_map_object(rb_id, SemanticMapLayer.ROADBLOCK)
            if roadblock is None:
                roadblock = map_api.get_map_object(rb_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            if roadblock is None:
                continue
            for lane in roadblock.interior_edges:
                points = []
                for state in lane.baseline_path.discrete_path:
                    lx, ly, l_heading = self._global_to_local(state.x, state.y, state.heading, current_ego_pose)
                    points.append([lx, ly, l_heading])
                if points:
                    reference_lines.append(np.array(points, dtype=np.float32))

        if not reference_lines:
            return {
                "position": torch.zeros((0, 0, 2), dtype=torch.float32),
                "vector": torch.zeros((0, 0, 2), dtype=torch.float32),
                "orientation": torch.zeros((0, 0), dtype=torch.float32),
                "valid_mask": torch.zeros((0, 0), dtype=torch.bool),
                "future_projection": torch.zeros((0, 8, 2), dtype=torch.float32),
            }

        n_points = int(self._map_radius / 1.0)
        position = np.zeros((len(reference_lines), n_points, 2), dtype=np.float32)
        vector = np.zeros((len(reference_lines), n_points, 2), dtype=np.float32)
        orientation = np.zeros((len(reference_lines), n_points), dtype=np.float32)
        valid_mask = np.zeros((len(reference_lines), n_points), dtype=bool)
        future_projection = np.zeros((len(reference_lines), 8, 2), dtype=np.float32)

        ego_future = scene.get_future_trajectory().poses
        if len(ego_future) > 0:
            linestrings = [LineString(line[:, :2]) for line in reference_lines]
            step = max(1, int(1.0 / NAVSIM_INTERVAL_LENGTH))
            future_samples = ego_future[step - 1 :: step]
            future_samples = [Point(xy) for xy in future_samples]

        for i, line in enumerate(reference_lines):
            subsample = line[::4][: n_points + 1]
            n_valid = len(subsample)
            if n_valid < 2:
                continue
            position[i, : n_valid - 1] = subsample[:-1, :2]
            vector[i, : n_valid - 1] = np.diff(subsample[:, :2], axis=0)
            orientation[i, : n_valid - 1] = subsample[:-1, 2]
            valid_mask[i, : n_valid - 1] = True

            if len(ego_future) > 0:
                for j, future_sample in enumerate(future_samples[:8]):
                    future_projection[i, j, 0] = linestrings[i].project(future_sample)
                    future_projection[i, j, 1] = linestrings[i].distance(future_sample)

        return {
            "position": torch.tensor(position, dtype=torch.float32),
            "vector": torch.tensor(vector, dtype=torch.float32),
            "orientation": torch.tensor(orientation, dtype=torch.float32),
            "valid_mask": torch.tensor(valid_mask, dtype=torch.bool),
            "future_projection": torch.tensor(future_projection, dtype=torch.float32),
        }
