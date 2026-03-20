### 1. **主要输入数据结构**

Pluto 模型的输入是一个字典，包含以下主要部分：

```python
data = {
    "agent": {...},           # 动态智能体信息
    "map": {...},            # 地图信息
    "static_objects": {...}, # 静态障碍物信息
    "reference_line": {...}, # 参考线信息（可选）
    "current_state": {...},  # 当前状态
    "causal": {...},         # 因果推理信息（训练时）
    "cost_maps": {...},      # 代价地图（训练时）
}
```

### 2. **详细输入说明**

#### **2.1 智能体数据 (`agent`)**
```python
agent = {
    "position":      (bs, A, T, 2)    # 位置坐标 (x, y)
    "heading":       (bs, A, T)       # 朝向角度
    "velocity":      (bs, A, T, 2)    # 速度向量 (vx, vy)
    "shape":         (bs, A, T, 2)    # 形状尺寸 (width, length)
    "category":      (bs, A)          # 类别编码 (0: ego, 1: vehicle, 2: pedestrian, 3: bicycle)
    "valid_mask":    (bs, A, T)       # 有效性掩码
    "target":        (bs, A, T_future, 3) # 未来轨迹目标 (位置+朝向)
}
```
- **A**: 智能体数量（包括自车ego）
- **T**: 历史时间步数（默认21步，2.1秒）
- **T_future**: 未来时间步数（默认80步，8秒）
- **索引0**: 总是自车（ego）

#### **2.2 地图数据 (`map`)**
```python
map = {
    "point_position":     (bs, M, 3, P, 2)    # 采样点位置 (中心线, 左边界, 右边界)
    "point_vector":       (bs, M, 3, P, 2)    # 采样点向量
    "point_orientation":  (bs, M, 3, P)       # 采样点朝向
    "point_side":         (bs, M, 3)          # 边界类型 (0:中心线, 1:左边界, 2:右边界)
    "polygon_center":     (bs, M, 3)          # 多边形中心 (x, y, heading)
    "polygon_position":   (bs, M, 2)          # 多边形起始位置
    "polygon_orientation":(bs, M)             # 多边形起始朝向
    "polygon_type":       (bs, M)             # 类型 (0: lane, 1: lane_connector, 2: crosswalk)
    "polygon_on_route":   (bs, M)             # 是否在规划路径上
    "polygon_tl_status":  (bs, M)             # 交通灯状态 (0:UNKNOWN, 1:GREEN, 2:YELLOW, 3:RED)
    "polygon_has_speed_limit": (bs, M)        # 是否有速度限制
    "polygon_speed_limit":(bs, M)             # 速度限制值
    "polygon_road_block_id": (bs, M)          # 道路块ID
    "valid_mask":         (bs, M, P)          # 有效性掩码
}
```
- **M**: 地图元素数量（车道线、连接线、人行横道）
- **P**: 每个元素的采样点数（默认20）

#### **2.3 静态障碍物 (`static_objects`)**
```python
static_objects = {
    "position":      (bs, N, 2)      # 位置坐标
    "heading":       (bs, N)         # 朝向角度
    "shape":         (bs, N, 2)      # 形状尺寸
    "category":      (bs, N)         # 类别编码 (0: CZONE_SIGN, 1: BARRIER, 2: TRAFFIC_CONE, 3: GENERIC_OBJECT)
    "valid_mask":    (bs, N)         # 有效性掩码
}
```
- **N**: 静态障碍物数量（最多10个）a