"""
单次重规划接口

用途:
- 将 `tentacle_local_planner_3.21.py` 中单次重规划的核心逻辑封装为可复用接口。
- 后续脚本只需要构造/加载 planner，并调用 `run_single_replan`，即可得到本次最优触须轨迹。

典型流程:
1) create_planner(...) 创建规划器
2) planner.load_global_trajectory(...), planner.load_environment(...)
3) run_single_replan(...)
"""

from __future__ import annotations

import importlib.util
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

OffsetSpec = Union[float, Tuple[float, float], List[float], np.ndarray]


@dataclass
class SingleReplanResult:
    """单次重规划结果。"""

    success: bool
    best_tentacle: Optional[Dict[str, Any]]
    speed_profile: Optional[Dict[str, Any]]  # 新增：速度规划结果
    tentacles: List[Dict[str, Any]]
    valid_count: int
    total_count: int
    phase: str
    current_time_index: int
    current_step: Optional[int]
    reference_segment: List[List[float]]
    timing: Dict[str, float]
    reason: str = ""
    trajectory: Optional[List[Dict[str, Any]]] = None


def _load_module_from_file(module_file: str):
    """从任意文件路径加载 Python 模块。"""
    module_file = os.path.abspath(module_file)
    if not os.path.exists(module_file):
        raise FileNotFoundError(f"planner file not found: {module_file}")

    module_name = os.path.splitext(os.path.basename(module_file))[0].replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from file: {module_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_planner(
    planner_file: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    num_trailers: int = 1,
):
    """
    创建 TentacleLocalPlanner 实例。

    Args:
        planner_file: 规划器文件路径。默认使用当前目录下 `tentacle_local_planner_3.21.py`。
        config: 传给 TentacleLocalPlanner 的配置字典。
        num_trailers: 挂车节数。
    """
    if planner_file is None:
        planner_file = os.path.join(os.path.dirname(__file__), "tentacle_local_planner_3.21.py")

    module = _load_module_from_file(planner_file)
    if not hasattr(module, "TentacleLocalPlanner"):
        raise AttributeError(f"TentacleLocalPlanner not found in {planner_file}")

    planner_cls = module.TentacleLocalPlanner
    return planner_cls(config=config, num_trailers=num_trailers)


def _build_reference_segment(planner, current_pos: np.ndarray) -> Tuple[List[List[float]], int]:
    """复用主流程里的参考段提取逻辑。"""
    if planner.global_trajectory is None:
        raise ValueError("planner.global_trajectory is None; please call load_global_trajectory first")

    lookahead_steps = int(float(planner.config["planning_horizon"]) / float(planner.config["dt"]))
    num_steps = len(planner.global_trajectory["step"])

    gx = np.asarray(planner.global_trajectory["tractor_x"], dtype=float)
    gy = np.asarray(planner.global_trajectory["tractor_y"], dtype=float)

    distances = np.hypot(gx - float(current_pos[0]), gy - float(current_pos[1]))
    current_step = int(np.argmin(distances))
    end_step = min(current_step + lookahead_steps, num_steps)

    reference_segment: List[List[float]] = []
    sample_interval = max(2, int(lookahead_steps / 50))
    for i in range(current_step, end_step, sample_interval):
        reference_segment.append([
            float(planner.global_trajectory["tractor_x"][i]),
            float(planner.global_trajectory["tractor_y"][i]),
        ])

    if len(reference_segment) > 0 and end_step > current_step:
        last_point = [
            float(planner.global_trajectory["tractor_x"][end_step - 1]),
            float(planner.global_trajectory["tractor_y"][end_step - 1]),
        ]
        if reference_segment[-1] != last_point:
            reference_segment.append(last_point)

    if (
        bool(planner.config.get("terminal_path_extension_enable", True))
        and end_step >= num_steps
        and len(reference_segment) >= 2
    ):
        reference_segment = planner._extend_reference_segment_beyond_goal(
            reference_segment,
            float(planner.config.get("terminal_path_extension_m", 0.0)),
        )

    return reference_segment, current_step


def _build_fallback_offsets_2d(
    primary_offsets: Sequence[float],
    fallback_n: int,
    max_lat: float,
) -> List[Tuple[float, float]]:
    """复用主流程中的 fallback 二维偏移采样。"""
    max_lat = float(max(0.0, max_lat))
    dl_levels_5 = list(np.linspace(-max_lat, max_lat, 5))
    ds_levels = [-4.0, 4.0, 0.0]
    candidates = [(ds, dl) for ds in ds_levels for dl in dl_levels_5]

    eps = 1e-6
    primary_set = [float(v) for v in list(primary_offsets)]

    def _is_primary(ds: float, dl: float) -> bool:
        if abs(float(ds)) > eps:
            return False
        return any(abs(float(dl) - pv) <= 1e-4 for pv in primary_set)

    out: List[Tuple[float, float]] = []
    seen = set()
    for ds, dl in candidates:
        if _is_primary(ds, dl):
            continue
        key = (round(float(ds), 4), round(float(dl), 4))
        if key in seen:
            continue
        seen.add(key)
        out.append((float(ds), float(dl)))

    if len(out) < int(fallback_n):
        extra_ds = [-2.0, 2.0, -6.0, 6.0]
        extra_dl = list(np.linspace(-max_lat, max_lat, 7))
        for ds in extra_ds:
            for dl in extra_dl:
                if _is_primary(ds, dl):
                    continue
                key = (round(float(ds), 4), round(float(dl), 4))
                if key in seen:
                    continue
                seen.add(key)
                out.append((float(ds), float(dl)))
                if len(out) >= int(fallback_n):
                    return out[: int(fallback_n)]

    return out[: int(fallback_n)]


def _compute_arc_length_xy(xy_points):
    """计算 XY 轨迹的累计弧长。"""
    if xy_points is None:
        return np.asarray([0.0], dtype=float)

    points = np.asarray(xy_points, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0:
        return np.asarray([0.0], dtype=float)
    if points.shape[0] == 1:
        return np.asarray([0.0], dtype=float)

    segment_lengths = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def _project_xy_to_path_s_idx_dist(path_xy, path_s, x, y):
    """将点投影到路径上，返回 (s, idx, lateral_dist)。"""
    if path_xy is None:
        return 0.0, 0, float(math.hypot(float(x), float(y)))

    points = np.asarray(path_xy, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0:
        return 0.0, 0, float(math.hypot(float(x), float(y)))
    if points.shape[0] == 1:
        dx = float(x) - float(points[0, 0])
        dy = float(y) - float(points[0, 1])
        return 0.0, 0, float(math.hypot(dx, dy))

    s_values = np.asarray(path_s, dtype=float) if path_s is not None else _compute_arc_length_xy(points)
    query = np.array([float(x), float(y)], dtype=float)

    best_idx = 0
    best_dist_sq = float('inf')
    best_proj_s = 0.0

    for i in range(points.shape[0] - 1):
        p0 = points[i]
        p1 = points[i + 1]
        seg = p1 - p0
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq <= 1e-12:
            continue
        t = float(np.dot(query - p0, seg) / seg_len_sq)
        t = float(max(0.0, min(1.0, t)))
        proj = p0 + t * seg
        d2 = float(np.dot(query - proj, query - proj))
        if d2 < best_dist_sq:
            best_dist_sq = d2
            best_idx = i
            best_proj_s = float(s_values[i] + t * (s_values[i + 1] - s_values[i]))

    if not np.isfinite(best_dist_sq):
        best_idx = 0
        best_proj_s = float(s_values[0]) if len(s_values) > 0 else 0.0
        dx = float(query[0] - points[0, 0])
        dy = float(query[1] - points[0, 1])
        return best_proj_s, best_idx, float(math.hypot(dx, dy))

    return best_proj_s, best_idx, float(math.sqrt(best_dist_sq))


def _dynamic_obstacle_state_at_time(obs, t):
    """获取动态障碍物在指定时刻的状态。"""
    times = np.asarray(obs.get('time', []), dtype=float)
    xs = np.asarray(obs.get('x', []), dtype=float)
    ys = np.asarray(obs.get('y', []), dtype=float)
    yaws = np.asarray(obs.get('yaw', []), dtype=float)
    vels = np.asarray(obs.get('velocity', []), dtype=float)

    if times.size == 0 or xs.size == 0 or ys.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    if times.size == 1 or xs.size == 1 or ys.size == 1:
        yaw = float(yaws[0]) if yaws.size > 0 else 0.0
        vel = float(vels[0]) if vels.size > 0 else 0.0
        return float(xs[0]), float(ys[0]), yaw, vel

    t = float(t)
    if t <= float(times[0]):
        idx = 0
        alpha = 0.0
    elif t >= float(times[-1]):
        idx = int(times.size - 2)
        alpha = 1.0
    else:
        idx = int(np.searchsorted(times, t, side='right') - 1)
        idx = max(0, min(idx, int(times.size - 2)))
        t0 = float(times[idx])
        t1 = float(times[idx + 1])
        alpha = 0.0 if abs(t1 - t0) < 1e-9 else (t - t0) / (t1 - t0)

    x = (1.0 - alpha) * float(xs[idx]) + alpha * float(xs[min(idx + 1, xs.size - 1)])
    y = (1.0 - alpha) * float(ys[idx]) + alpha * float(ys[min(idx + 1, ys.size - 1)])
    yaw = float((1.0 - alpha) * float(yaws[idx if yaws.size > idx else -1]) + alpha * float(yaws[min(idx + 1, yaws.size - 1)])) if yaws.size > 0 else 0.0
    vel = float((1.0 - alpha) * float(vels[idx if vels.size > idx else -1]) + alpha * float(vels[min(idx + 1, vels.size - 1)])) if vels.size > 0 else 0.0
    return x, y, yaw, vel


def _shape_speed_profile(planner, profile, current_speed):
    """对最终速度轨迹做连续化整形，减少相邻重规划拼接造成的尖角。"""
    if not profile:
        return profile

    t_list = list(profile.get('t', []))
    v_raw = np.asarray(profile.get('v', []), dtype=float)
    if len(t_list) != len(v_raw) or len(v_raw) <= 1:
        return profile

    dt = float(planner.config['dt'])
    a_min = float(planner.config['a_min'])
    a_max = float(planner.config['a_max'])
    v_max = float(planner.config['v_max'])
    alpha_up = float(planner.config.get('speed_shape_alpha_up', 0.45))
    alpha_down = float(planner.config.get('speed_shape_alpha_down', 0.70))
    jerk_lim = float(planner.config.get('speed_shape_jerk_limit', 8.0))

    v_shaped = np.zeros_like(v_raw)
    v_shaped[0] = float(max(0.0, current_speed))
    prev_a = float(getattr(planner, 'last_exec_acc', 0.0))

    for i in range(1, len(v_raw)):
        v_prev = float(v_shaped[i - 1])
        v_ref = float(v_raw[i])
        alpha = alpha_up if v_ref >= v_prev else alpha_down
        v_target = (1.0 - alpha) * v_prev + alpha * v_ref

        a_des = (v_target - v_prev) / dt
        a_low = max(a_min, prev_a - jerk_lim * dt)
        a_high = min(a_max, prev_a + jerk_lim * dt)
        a_cmd = float(max(a_low, min(a_high, a_des)))
        v_next = float(max(0.0, min(v_max, v_prev + a_cmd * dt)))

        if v_ref >= v_prev:
            v_next = min(v_next, v_ref)

        v_shaped[i] = v_next
        prev_a = a_cmd

    s_shaped = np.zeros_like(v_shaped)
    for i in range(1, len(v_shaped)):
        s_shaped[i] = s_shaped[i - 1] + 0.5 * (v_shaped[i - 1] + v_shaped[i]) * dt

    s_raw = np.asarray(profile.get('s', []), dtype=float)
    if len(s_raw) == len(s_shaped) and len(s_raw) > 0:
        s_cap = float(s_raw[-1])
        s_shaped = np.minimum(s_shaped, s_cap)

    out = dict(profile)
    out['v'] = [float(v) for v in v_shaped]
    out['s'] = [float(s) for s in s_shaped]
    if len(v_shaped) >= 2:
        out['a_exec'] = float((v_shaped[1] - v_shaped[0]) / dt)
    else:
        out['a_exec'] = float(profile.get('a_exec', profile.get('a', 0.0)))
    return out


def plan_speed_profile(planner, best_tentacle, current_speed, current_time):
    """独立纵向速度规划：不再依赖 TentacleLocalPlanner 的成员方法。"""
    dt = float(planner.config['dt'])
    planning_horizon = float(planner.config['planning_horizon'])

    if best_tentacle is None:
        emergency_a = 2.0 * float(planner.config['a_min'])
        horizon_steps = max(2, int(round(planning_horizon / dt)))
        s_list = [0.0]
        v_list = [float(max(0.0, current_speed))]
        t_list = [0.0]
        for k in range(1, horizon_steps):
            t_k = k * dt
            v_next = max(0.0, v_list[-1] + emergency_a * dt)
            s_next = s_list[-1] + v_list[-1] * dt + 0.5 * emergency_a * dt * dt
            s_list.append(float(max(0.0, min(s_next, s_end))))
            v_list.append(float(v_next))
            t_list.append(t_k)
            if v_next <= 1e-3:
                break
        return _shape_speed_profile(planner, {'t': t_list, 's': s_list, 'v': v_list, 'a': emergency_a, 'a_exec': emergency_a, 'path_s': np.asarray([0.0], dtype=float)}, current_speed)

    traj = np.array(best_tentacle['tractor_trajectory'], dtype=float)
    if traj.ndim != 2 or traj.shape[0] == 0:
        return _shape_speed_profile(planner, {'t': [0.0], 's': [0.0], 'v': [float(max(0.0, current_speed))], 'a': 0.0, 'a_exec': 0.0, 'path_s': np.asarray([0.0], dtype=float)}, current_speed)

    path_xy = traj[:, :2]
    path_s = _compute_arc_length_xy(path_xy)
    s_end = float(path_s[-1]) if len(path_s) > 0 else 0.0

    v_max_config = float(planner.config['v_max'])
    a_max = float(planner.config['a_max'])
    a_min = float(planner.config['a_min'])

    a_lat_max = float(planner.config.get('a_lat_max', 4.5))
    max_kappa = 1e-6
    if len(path_xy) >= 3:
        for i in range(1, len(path_xy) - 1):
            p1, p2, p3 = path_xy[i - 1], path_xy[i], path_xy[i + 1]
            a_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            b_len = math.hypot(p3[0] - p2[0], p3[1] - p2[1])
            c_len = math.hypot(p3[0] - p1[0], p3[1] - p1[1])
            if a_len > 0 and b_len > 0 and c_len > 0:
                s_p = (a_len + b_len + c_len) / 2.0
                area_sq = max(0.0, s_p * (s_p - a_len) * (s_p - b_len) * (s_p - c_len))
                if area_sq > 0:
                    kappa = 4.0 * math.sqrt(area_sq) / (a_len * b_len * c_len)
                    max_kappa = max(max_kappa, kappa)
    v_max_kappa = max(1.8, math.sqrt(a_lat_max / max_kappa))
    v_max = min(v_max_config, v_max_kappa)

    acc_candidates = [a_min, -1.5, -1.0, -0.5, 0.0, 0.3, a_max]
    acc_candidates = sorted({float(max(a_min, min(a_max, a))) for a in acc_candidates})

    follow_headway = float(max(0.2, planner.config.get('follow_headway', 0.70)))
    follow_min_gap = float(max(0.0, planner.config.get('follow_min_gap', 0.80)))
    follow_gain = float(max(0.0, planner.config.get('follow_gain', 0.55)))
    safety_margin = float(max(0.0, planner.config.get('safety_margin', 2.0)))
    follow_heading_th = math.radians(35.0)

    best = None
    best_cost = float('inf')

    w_progress = float(planner.config.get('w_progress', 1.0))
    w_acc = float(planner.config.get('w_acc', 0.2))
    w_speed = float(planner.config.get('w_speed', 0.5))
    w_smooth = float(planner.config.get('w_smooth', 1.0))
    w_safety = float(planner.config.get('w_safety', 10.0))

    v_target_base = float(max(0.0, min(v_max, v_max)))
    prev_a = float(getattr(planner, 'last_exec_acc', 0.0))

    min_horizon_s = float(max(2.0 * dt, 1.5))
    if s_end > 1e-6:
        horizon_ref_speed = float(max(1.0, min(v_max, max(float(current_speed), 0.9 * v_target_base))))
        cover_time = float(s_end / horizon_ref_speed)
        planning_horizon_s = float(min(planning_horizon, max(min_horizon_s, 1.15 * cover_time)))
    else:
        planning_horizon_s = min_horizon_s

    horizon_steps = max(2, int(round(planning_horizon_s / dt)))
    T1 = 1.0
    blend_time = 0.80
    half_blend = 0.5 * blend_time

    valid_obstacles = []
    dynamic_obstacles = getattr(planner, 'dynamic_obstacles', None) or []
    if dynamic_obstacles:
        for obs in dynamic_obstacles:
            ox, oy, _oyaw, _ov = _dynamic_obstacle_state_at_time(obs, current_time)
            _, _, lateral_dist = _project_xy_to_path_s_idx_dist(path_xy, path_s, ox, oy)
            obs_w = float(obs.get('width', 0.0))
            gate_th = 0.5 * (float(planner.tractor_width) + obs_w) + safety_margin
            if lateral_dist <= gate_th:
                valid_obstacles.append(obs)

    for a1 in acc_candidates:
        for a2 in acc_candidates:
            a1 = float(max(a_min, min(a_max, a1)))
            a2 = float(max(a_min, min(a_max, a2)))

            s_list = [0.0]
            v_list = [float(max(0.0, current_speed))]
            t_list = [0.0]

            feasible = True
            penalty_cost = 0.0
            speed_error_accum = 0.0
            speed_error_count = 0

            for k in range(1, horizon_steps):
                t_k = k * dt
                if t_k <= (T1 - half_blend):
                    current_a = a1
                elif t_k >= (T1 + half_blend):
                    current_a = a2
                else:
                    ratio = (t_k - (T1 - half_blend)) / max(dt, blend_time)
                    current_a = (1.0 - ratio) * a1 + ratio * a2

                v_next = float(v_list[-1] + current_a * dt)
                if v_next > v_max:
                    v_next = min(v_next, max(v_max, v_list[-1] + a_min * dt))
                v_next = float(max(0.0, v_next))

                s_next = float(s_list[-1] + v_list[-1] * dt + 0.5 * current_a * dt * dt)
                s_next = float(max(0.0, min(s_next, s_end)))

                step_target_v = v_target_base
                for obs in valid_obstacles:
                    ox, oy, obs_yaw, ov = _dynamic_obstacle_state_at_time(obs, current_time + t_k)
                    obs_s, obs_idx, lateral_dist = _project_xy_to_path_s_idx_dist(path_xy, path_s, ox, oy)
                    obs_l = float(obs.get('length', 0.0))
                    obs_w = float(obs.get('width', 0.0))

                    gate_th = 0.5 * (float(planner.tractor_width) + obs_w) + safety_margin
                    if lateral_dist > gate_th:
                        continue
                    if obs_s > (s_end + safety_margin):
                        continue

                    geom_dist = 0.5 * (float(planner.tractor_length) + obs_l)
                    if len(path_xy) >= 2:
                        idx0 = max(0, min(obs_idx, len(path_xy) - 1))
                        idx1 = min(idx0 + 1, len(path_xy) - 1)
                        if idx1 == idx0 and idx0 > 0:
                            idx1 = idx0 - 1
                        p0 = path_xy[idx0]
                        p1 = path_xy[idx1]
                        path_yaw = float(math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0])))
                    else:
                        path_yaw = 0.0
                    heading_diff = abs(math.atan2(math.sin(obs_yaw - path_yaw), math.cos(obs_yaw - path_yaw)))
                    same_direction = heading_diff <= follow_heading_th

                    collision_threshold = geom_dist + 0.2
                    desired_gap = geom_dist + follow_min_gap + follow_headway * max(v_next, float(ov))
                    safety_threshold = geom_dist + safety_margin + follow_headway * max(v_next, float(ov))

                    dist_s = obs_s - s_next
                    abs_dist = abs(dist_s)

                    if abs_dist < collision_threshold:
                        feasible = False
                        break

                    if dist_s > 0 and same_direction:
                        follow_v = float(ov) + follow_gain * float(dist_s - desired_gap)
                        follow_v = float(max(0.0, min(v_target_base, follow_v)))
                        step_target_v = min(step_target_v, follow_v)
                        if dist_s < desired_gap:
                            penalty_cost += (desired_gap - dist_s) ** 2
                        if dist_s < safety_threshold:
                            penalty_cost += 4.0 * (safety_threshold - dist_s) ** 2
                    elif dist_s > 0 and not same_direction:
                        if abs_dist < safety_threshold:
                            penalty_cost += 2.0 * (safety_threshold - abs_dist) ** 2
                    elif dist_s <= 0 and abs_dist < safety_threshold:
                        penalty_cost += 4.0 * (safety_threshold - abs_dist) ** 2

                if not feasible:
                    break

                speed_error_accum += float((step_target_v - v_next) ** 2)
                speed_error_count += 1

                s_list.append(s_next)
                v_list.append(v_next)
                t_list.append(t_k)

                if s_next >= s_end - 1e-6:
                    break

            if not feasible:
                continue

            s_final = s_list[-1]
            T = float(t_list[-1]) if t_list else float(len(s_list)) * dt
            if speed_error_count > 0:
                speed_term = float(speed_error_accum / speed_error_count)
            else:
                v_mean = float(np.mean(v_list)) if len(v_list) > 0 else 0.0
                speed_term = float((v_target_base - v_mean) ** 2)

            jerk_term = abs(a2 - a1) / max(blend_time, dt)
            acc_change_term = (a1 - prev_a) ** 2
            smooth_term = jerk_term + acc_change_term

            cost = (
                -w_progress * float(s_final) * 10.0
                + w_acc * (float(a1 * a1) + float(a2 * a2)) * T * 0.5
                + w_speed * speed_term * 20.0
                + w_smooth * smooth_term
                + w_safety * penalty_cost
            )

            if cost < best_cost:
                best_cost = cost
                a_exec = a1
                if len(v_list) >= 2:
                    a_exec = (float(v_list[1]) - float(v_list[0])) / dt
                best = {
                    't': t_list,
                    's': s_list,
                    'v': v_list,
                    'a': a1,
                    'a_exec': float(a_exec),
                    'path_s': path_s,
                }

    if best is None:
        emergency_a = 2.0 * a_min
        s_list = [0.0]
        v_list = [float(max(0.0, current_speed))]
        t_list = [0.0]
        for k in range(1, horizon_steps):
            t_k = k * dt
            v_next = max(0.0, v_list[-1] + emergency_a * dt)
            s_next = s_list[-1] + v_list[-1] * dt + 0.5 * emergency_a * dt * dt
            s_list.append(float(max(0.0, min(s_next, s_end))))
            v_list.append(float(v_next))
            t_list.append(t_k)
            if v_next <= 1e-3 or s_next >= s_end - 1e-6:
                break
        best = {
            't': t_list,
            's': s_list,
            'v': v_list,
            'a': emergency_a,
            'a_exec': float(emergency_a),
            'path_s': path_s,
        }

    return _shape_speed_profile(planner, best, current_speed)


def _interpolate_trajectory_by_s(path_xy, path_s, target_s):
    """在路径上根据弧长 s 进行插值，返回 (x, y)。"""
    if not isinstance(path_xy, np.ndarray):
        path_xy = np.array(path_xy)
    if not isinstance(path_s, np.ndarray):
        path_s = np.array(path_s)

    if target_s <= path_s[0]:
        return path_xy[0]
    if target_s >= path_s[-1]:
        return path_xy[-1]

    idx = np.searchsorted(path_s, target_s, side='right') - 1
    s0, s1 = path_s[idx], path_s[idx + 1]
    p0, p1 = path_xy[idx], path_xy[idx + 1]

    if abs(s1 - s0) < 1e-6:
        return p0

    ratio = (target_s - s0) / (s1 - s0)
    return p0 + ratio * (p1 - p0)


def _angle_lerp(a0, a1, t):
    da = math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))
    return a0 + t * da


def _find_segment_by_s(s_points, s_query):
    """返回 (i0, i1, t) 使得 s_query 在 [s[i0], s[i1]] 内。"""
    if s_points is None or len(s_points) == 0:
        return 0, 0, 0.0
    if s_query <= s_points[0]:
        return 0, 0, 0.0
    if s_query >= s_points[-1]:
        last = len(s_points) - 1
        return last, last, 0.0

    idx = int(np.searchsorted(s_points, s_query, side='right'))
    i1 = max(1, min(idx, len(s_points) - 1))
    i0 = i1 - 1
    ds = float(s_points[i1] - s_points[i0])
    t = 0.0 if ds <= 1e-9 else float((s_query - s_points[i0]) / ds)
    t = max(0.0, min(1.0, t))
    return i0, i1, t


def _interp_traj_by_index(traj, i0, i1, t):
    """按离散索引 (i0,i1,t) 在轨迹上插值，支持 yaw 角插值。"""
    i0 = int(max(0, min(i0, len(traj) - 1)))
    i1 = int(max(0, min(i1, len(traj) - 1)))
    t = float(max(0.0, min(1.0, t)))

    p0 = traj[i0]
    p1 = traj[i1]
    x = (1.0 - t) * float(p0[0]) + t * float(p1[0])
    y = (1.0 - t) * float(p0[1]) + t * float(p1[1])
    if len(p0) >= 3 and len(p1) >= 3:
        yaw = _angle_lerp(float(p0[2]), float(p1[2]), t)
    else:
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        yaw = float(math.atan2(dy, dx)) if (abs(dx) + abs(dy)) > 1e-12 else 0.0

    return np.array([x, y, yaw], dtype=float)


def _build_combined_trajectory(planner, best_tentacle, speed_profile):
    """将路径和速度规划结合成完整的时空轨迹。"""
    if not best_tentacle or not speed_profile or 's' not in speed_profile:
        return None

    tractor_traj = best_tentacle.get('tractor_trajectory', []) or []
    if len(tractor_traj) < 2:
        return None

    path_xy = np.array(tractor_traj, dtype=float)[:, :2]
    path_s_geom = _compute_arc_length_xy(path_xy)
    if len(path_s_geom) < 2:
        return None

    trailer_trajs = best_tentacle.get('trailer_trajectories', []) or []

    s_profile = speed_profile['s']
    v_profile = speed_profile['v']
    t_profile = speed_profile['t']

    combined_traj = []
    for i in range(len(t_profile)):
        s = float(s_profile[i])
        v = float(v_profile[i])
        t = float(t_profile[i])

        i0, i1, alpha = _find_segment_by_s(path_s_geom, s)
        tractor_point = _interp_traj_by_index(tractor_traj, i0, i1, alpha)
        x = float(tractor_point[0])
        y = float(tractor_point[1])
        yaw = float(tractor_point[2])

        point_data = {
            't': t, 's': s, 'v': v, 'x': x, 'y': y, 'yaw': yaw,
        }

        fallback_states = None
        for trailer_idx in range(planner.num_trailers):
            if trailer_idx < len(trailer_trajs) and len(trailer_trajs[trailer_idx]) > 0:
                trailer_point = _interp_traj_by_index(trailer_trajs[trailer_idx], i0, i1, alpha)
                tx = float(trailer_point[0])
                ty = float(trailer_point[1])
                tyaw = float(trailer_point[2])
            else:
                if fallback_states is None:
                    tractor_state = {'x': x, 'y': y, 'yaw': yaw}
                    fallback_states = planner.kinematics.initialize_trailers(tractor_state)
                if trailer_idx < len(fallback_states):
                    tx = float(fallback_states[trailer_idx]['x'])
                    ty = float(fallback_states[trailer_idx]['y'])
                    tyaw = float(fallback_states[trailer_idx]['yaw'])
                else:
                    tx, ty, tyaw = x, y, yaw

            point_data[f'trailer{trailer_idx+1}_x'] = tx
            point_data[f'trailer{trailer_idx+1}_y'] = ty
            point_data[f'trailer{trailer_idx+1}_yaw'] = tyaw

        combined_traj.append(point_data)

    return combined_traj


def build_speed_and_trajectory(planner, best_tentacle, current_speed, elapsed_time):
    """基于给定触须重算速度剖面并构建时空轨迹。"""
    if not best_tentacle:
        return None, None

    speed_profile = plan_speed_profile(
        planner, best_tentacle, float(current_speed), float(elapsed_time)
    )
    combined_trajectory = _build_combined_trajectory(planner, best_tentacle, speed_profile)
    best_tentacle['speed_profile'] = speed_profile
    best_tentacle['trajectory'] = combined_trajectory
    return speed_profile, combined_trajectory


def _evaluate_tentacles(planner, tentacles, current_time_index):
    """
    评估所有触须的代价
    """
    if not tentacles:
        return 0
    
    valid_count = 0
    for tentacle in tentacles:
        tentacle['cost'] = evaluate_tentacle_cost(planner, tentacle, current_time_index)
        if tentacle.get('is_valid', True):
            valid_count += 1
            
    return valid_count


def _generate_and_eval(
    planner,
    current_pos: np.ndarray,
    current_yaw: float,
    current_trailer_states: List[Dict[str, Any]],
    reference_segment: Sequence[Sequence[float]],
    lateral_offsets: Sequence[OffsetSpec],
    current_time_index: int,
):
    """
    生成并评估所有触须
    """
    tentacles = generate_tentacles(
        planner, current_pos, current_yaw, reference_segment, current_trailer_states, lateral_offsets
    )
    valid_count = _evaluate_tentacles(planner, tentacles, current_time_index)
    return tentacles, valid_count


# =====================================================================================
#  The following functions are migrated from tentacle_local_planner_3.21.py
#  to make this interface self-contained.
# =====================================================================================

def _generate_quintic_polynomial_trajectory(start_pos, start_yaw, start_curvature,
                                             end_pos, end_yaw, end_curvature, distance):
    """
    使用五次多项式生成平滑轨迹
    """
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    
    cos_yaw = np.cos(start_yaw)
    sin_yaw = np.sin(start_yaw)
    
    s_end = float(distance) if distance is not None else (dx * cos_yaw + dy * sin_yaw)
    d_end = -dx * sin_yaw + dy * cos_yaw
    
    if abs(s_end) < 1e-6:
        return None
    
    a0 = 0.0
    a1 = 0.0
    a2 = 0.0
    
    d_prime_end = np.tan(end_yaw - start_yaw) if abs(end_yaw - start_yaw) < np.pi/3 else 0.0
    
    s = s_end
    A = np.array([
        [s**3, s**4, s**5],
        [3*s**2, 4*s**3, 5*s**4],
        [6*s, 12*s**2, 20*s**3]
    ])
    b = np.array([d_end, d_prime_end, 0.0])
    
    try:
        coeffs = np.linalg.solve(A, b)
        a3, a4, a5 = coeffs
    except:
        return None
    
    tractor_trajectory = []
    num_points = max(int(distance / 0.5), 10)
    
    for i in range(num_points + 1):
        s = i * s_end / num_points
        d = a0 + a1*s + a2*s**2 + a3*s**3 + a4*s**4 + a5*s**5
        d_prime = a1 + 2*a2*s + 3*a3*s**2 + 4*a4*s**3 + 5*a5*s**4
        
        x_local = s
        y_local = d
        
        x_global = start_pos[0] + x_local * cos_yaw - y_local * sin_yaw
        y_global = start_pos[1] + x_local * sin_yaw + y_local * cos_yaw
        
        yaw_global = start_yaw + np.arctan2(d_prime, 1.0)
        
        tractor_trajectory.append([x_global, y_global, yaw_global])
    
    return tractor_trajectory


def _generate_trailer_trajectories_geometric(kinematics, start_trailer_states, tractor_trajectory):
    """
    基于牵引车轨迹生成挂车轨迹 (纯几何重构版本)
    """
    num_trailers = len(start_trailer_states)
    trailer_trajectories = [[] for _ in range(num_trailers)]
    
    trailer_states = [ts.copy() for ts in start_trailer_states]
    
    for i in range(num_trailers):
        trailer_trajectories[i].append([
            trailer_states[i]['x'],
            trailer_states[i]['y'],
            trailer_states[i]['yaw']
        ])
    
    for i in range(1, len(tractor_trajectory)):
        curr_tractor_point = tractor_trajectory[i]
        curr_tractor_yaw = float(curr_tractor_point[2])
        
        for j in range(num_trailers):
            if j == 0:
                hitch_x = curr_tractor_point[0] - kinematics.params.Lb * math.cos(curr_tractor_yaw)
                hitch_y = curr_tractor_point[1] - kinematics.params.Lb * math.sin(curr_tractor_yaw)
                
                prev_x = trailer_states[j]['x']
                prev_y = trailer_states[j]['y']
                
                dx = prev_x - hitch_x
                dy = prev_y - hitch_y
                
                dist = math.hypot(dx, dy)
                
                if dist > 1e-6:
                    new_yaw = math.atan2(dy, dx)
                    new_x = hitch_x + kinematics.params.trailer_L[j] * math.cos(new_yaw)
                    new_y = hitch_y + kinematics.params.trailer_L[j] * math.sin(new_yaw)
                else:
                    new_x = prev_x
                    new_y = prev_y
                    new_yaw = trailer_states[j]['yaw']
                
                alpha = 0.7
                smooth_x = (1 - alpha) * prev_x + alpha * new_x
                smooth_y = (1 - alpha) * prev_y + alpha * new_y
                
                dx = hitch_x - smooth_x
                dy = hitch_y - smooth_y
                smooth_yaw = math.atan2(dy, dx)
                
                trailer_states[j]['x'] = smooth_x
                trailer_states[j]['y'] = smooth_y
                trailer_states[j]['yaw'] = smooth_yaw
            else:
                prev_trailer_state = trailer_states[j - 1]
                
                hitch_x = prev_trailer_state['x'] - kinematics.params.trailer_Lb[j-1] * math.cos(prev_trailer_state['yaw'])
                hitch_y = prev_trailer_state['y'] - kinematics.params.trailer_Lb[j-1] * math.sin(prev_trailer_state['yaw'])
                
                prev_x = trailer_states[j]['x']
                prev_y = trailer_states[j]['y']
                
                dx = prev_x - hitch_x
                dy = prev_y - hitch_y
                dist = math.hypot(dx, dy)
                
                if dist > 1e-6:
                    new_yaw = math.atan2(dy, dx)
                    new_x = hitch_x + kinematics.params.trailer_L[j] * math.cos(new_yaw)
                    new_y = hitch_y + kinematics.params.trailer_L[j] * math.sin(new_yaw)
                else:
                    new_x = prev_x
                    new_y = prev_y
                    new_yaw = trailer_states[j]['yaw']
                
                alpha = 0.7
                smooth_x = (1 - alpha) * prev_x + alpha * new_x
                smooth_y = (1 - alpha) * prev_y + alpha * new_y
                
                dx = hitch_x - smooth_x
                dy = hitch_y - smooth_y
                smooth_yaw = math.atan2(dy, dx)
                
                trailer_states[j]['x'] = smooth_x
                trailer_states[j]['y'] = smooth_y
                trailer_states[j]['yaw'] = smooth_yaw
        
        for j in range(num_trailers):
            trailer_trajectories[j].append([
                trailer_states[j]['x'],
                trailer_states[j]['y'],
                trailer_states[j]['yaw']
            ])
    
    return trailer_trajectories


def _generate_reference_offset_trajectory(current_pos, current_yaw,
                                           reference_path_segment, longitudinal_distance,
                                           ds, dl):
    """
    沿参考线生成偏移触须
    """
    if reference_path_segment is None or len(reference_path_segment) < 2:
        return None

    seg_lens = []
    total_len = 0.0
    for i in range(len(reference_path_segment) - 1):
        dx = reference_path_segment[i + 1][0] - reference_path_segment[i][0]
        dy = reference_path_segment[i + 1][1] - reference_path_segment[i][1]
        l = float(math.hypot(dx, dy))
        seg_lens.append(l)
        total_len += l

    if total_len < 1e-6:
        return None

    target_len = float(min(total_len, max(0.0, longitudinal_distance)))
    if target_len < 1e-6:
        return None

    ref_points = [np.array(reference_path_segment[0], dtype=float)]
    acc = 0.0
    for i, l in enumerate(seg_lens):
        p0 = np.array(reference_path_segment[i], dtype=float)
        p1 = np.array(reference_path_segment[i + 1], dtype=float)
        if acc + l < target_len - 1e-6:
            ref_points.append(p1)
            acc += l
        else:
            t = (target_len - acc) / max(l, 1e-9)
            p = p0 + t * (p1 - p0)
            ref_points.append(p)
            acc += l
            break

    if len(ref_points) < 2:
        return None

    p0 = ref_points[0]
    p1 = ref_points[1]
    dx0 = p1[0] - p0[0]
    dy0 = p1[1] - p0[1]
    yaw0 = math.atan2(dy0, dx0)
    tx0 = math.cos(yaw0)
    ty0 = math.sin(yaw0)
    nx0 = -ty0
    ny0 = tx0
    
    diff_x = current_pos[0] - p0[0]
    diff_y = current_pos[1] - p0[1]
    current_ds = diff_x * tx0 + diff_y * ty0
    current_dl = diff_x * nx0 + diff_y * ny0

    num_pts = len(ref_points)
    trajectory_xy = []
    for idx, p in enumerate(ref_points):
        t = float(idx) / float(num_pts - 1) if num_pts > 1 else 1.0
        blend = 6 * (t**5) - 15 * (t**4) + 10 * (t**3)
        
        ds_i = current_ds + blend * (float(ds) - current_ds)
        dl_i = current_dl + blend * (float(dl) - current_dl)

        if idx == 0:
            p_next = ref_points[1]
            dx = p_next[0] - p[0]
            dy = p_next[1] - p[1]
        elif idx == len(ref_points) - 1:
            p_prev = ref_points[-2]
            dx = p[0] - p_prev[0]
            dy = p[1] - p_prev[1]
        else:
            p_prev = ref_points[idx - 1]
            p_next = ref_points[idx + 1]
            dx = p_next[0] - p_prev[0]
            dy = p_next[1] - p_prev[1]

        if abs(dx) + abs(dy) < 1e-9:
            yaw = float(current_yaw)
        else:
            yaw = float(math.atan2(dy, dx))

        tx = float(math.cos(yaw))
        ty = float(math.sin(yaw))
        nx = -ty
        ny = tx

        x = float(p[0]) + ds_i * tx + dl_i * nx
        y = float(p[1]) + ds_i * ty + dl_i * ny
        trajectory_xy.append([x, y])

    trajectory = []
    for idx in range(num_pts):
        x, y = trajectory_xy[idx]
        if idx == 0:
            yaw = float(current_yaw)
        elif idx == num_pts - 1:
            dx = x - trajectory_xy[idx - 1][0]
            dy = y - trajectory_xy[idx - 1][1]
            yaw = float(math.atan2(dy, dx)) if (abs(dx)+abs(dy)) > 1e-9 else float(current_yaw)
        else:
            dx = trajectory_xy[idx + 1][0] - trajectory_xy[idx - 1][0]
            dy = trajectory_xy[idx + 1][1] - trajectory_xy[idx - 1][1]
            yaw = float(math.atan2(dy, dx)) if (abs(dx)+abs(dy)) > 1e-9 else float(current_yaw)
        trajectory.append([x, y, yaw])

    trajectory[0][0] = float(current_pos[0])
    trajectory[0][1] = float(current_pos[1])
    trajectory[0][2] = float(current_yaw)
    return trajectory


def generate_tentacles(
    planner, current_pos, current_yaw, reference_path_segment, trailer_states=None, lateral_offsets=None
):
    """
    生成触须轨迹集合
    """
    tentacles = []
    
    if trailer_states is None:
        tractor_state = {'x': current_pos[0], 'y': current_pos[1], 'yaw': current_yaw}
        trailer_states = planner.kinematics.initialize_trailers(tractor_state)
    
    if len(reference_path_segment) < 2:
        return tentacles
    
    seg_lens = []
    total_len = 0.0
    for i in range(len(reference_path_segment) - 1):
        dx = reference_path_segment[i + 1][0] - reference_path_segment[i][0]
        dy = reference_path_segment[i + 1][1] - reference_path_segment[i][1]
        l = float(math.hypot(dx, dy))
        seg_lens.append(l)
        total_len += l

    desired_len = float(planner.config.get('tentacle_length_m', 0.0) or 0.0)
    longitudinal_distance = float(total_len)
    if desired_len > 1e-6:
        longitudinal_distance = float(min(longitudinal_distance, desired_len))

    end_pos = np.array(reference_path_segment[-1], dtype=float)
    end_yaw = float(
        math.atan2(
            reference_path_segment[-1][1] - reference_path_segment[-2][1],
            reference_path_segment[-1][0] - reference_path_segment[-2][0],
        )
    )
    if longitudinal_distance < total_len - 1e-6:
        acc = 0.0
        for i, l in enumerate(seg_lens):
            if acc + l >= longitudinal_distance:
                t = (longitudinal_distance - acc) / max(l, 1e-9)
                p0 = np.array(reference_path_segment[i], dtype=float)
                p1 = np.array(reference_path_segment[i + 1], dtype=float)
                end_pos = p0 + t * (p1 - p0)
                end_yaw = float(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
                break
            acc += l
    
    max_lateral = planner.config['lateral_deviation']
    
    if lateral_offsets is None:
        lateral_offsets = np.linspace(-max_lateral, max_lateral, planner.config['num_tentacles'])

    offset_specs = list(lateral_offsets)
    
    cy = float(math.sin(float(end_yaw)))
    cx = float(math.cos(float(end_yaw)))
    ny = float(math.sin(float(end_yaw + math.pi / 2.0)))
    nx = float(math.cos(float(end_yaw + math.pi / 2.0)))

    for spec in offset_specs:
        if isinstance(spec, (list, tuple, np.ndarray)) and len(spec) == 2:
            ds = float(spec[0])
            dl = float(spec[1])
        else:
            ds = 0.0
            dl = float(spec)

        tractor_trajectory = _generate_reference_offset_trajectory(
            current_pos,
            current_yaw,
            reference_path_segment,
            longitudinal_distance,
            ds,
            dl,
        )

        if tractor_trajectory is None:
            target_x = float(end_pos[0]) + ds * cx + dl * nx
            target_y = float(end_pos[1]) + ds * cy + dl * ny
            target_pos = np.array([target_x, target_y], dtype=float)
            tractor_trajectory = _generate_quintic_polynomial_trajectory(
                current_pos, current_yaw, 0.0,
                target_pos, end_yaw, 0.0,
                longitudinal_distance
            )
        
        if tractor_trajectory is not None:
            trailer_trajectories = _generate_trailer_trajectories_geometric(
                planner.kinematics, trailer_states, tractor_trajectory
            )
            
            tentacle = {
                'lateral_offset': dl,
                'longitudinal_offset': ds,
                'tractor_trajectory': tractor_trajectory,
                'trailer_trajectories': trailer_trajectories,
                'cost': 0.0,
                'is_valid': True
            }
            tentacles.append(tentacle)
    
    return tentacles


def _check_trajectory_collision(
    trajectory, obstacles, vehicle_length, vehicle_width, config, allow_hard_collision=True, num_circles=1
):
    """
    检查单个轨迹是否与障碍物碰撞
    """
    num_circles = int(max(1, num_circles))

    def _vehicle_circles(px, py, yaw):
        r = 0.5 * float(vehicle_width)
        if num_circles <= 1 or float(vehicle_length) <= 2.0 * r + 1e-9:
            return [(float(px), float(py), float(r))]
        half = 0.5 * float(vehicle_length) - r
        if half <= 1e-9:
            return [(float(px), float(py), float(r))]
        offs = np.linspace(-half, half, num_circles)
        cy = float(math.sin(float(yaw)))
        cx = float(math.cos(float(yaw)))
        out = []
        for off in offs:
            out.append((float(px) + float(off) * cx, float(py) + float(off) * cy, float(r)))
        return out

    collision_margin = float(config.get('collision_margin', config.get('safety_margin', 0.0)))
    extra_buffer = float(config.get('obstacle_safety_buffer', 0.2))
    extra_buffer = float(max(0.0, extra_buffer))
    hard_overlap_tol = float(config.get('collision_hard_overlap_tolerance', 0.0))
    hard_overlap_tol = float(max(0.0, hard_overlap_tol))

    worst_cost = 0.0
    hard_collision = False
    collision_info = ""
    
    for point in trajectory:
        x, y = float(point[0]), float(point[1])
        yaw = float(point[2]) if len(point) >= 3 else 0.0
        circles = _vehicle_circles(x, y, yaw)
        
        for obs in obstacles:
            ox = float(obs['x'])
            oy = float(obs['y'])
            l = float(obs.get('length', 0.0))
            w = float(obs.get('width', 0.0))
            yaw_obs = float(obs.get('yaw', 0.0))
            
            cos_yaw = math.cos(-yaw_obs)
            sin_yaw = math.sin(-yaw_obs)
            
            min_dist_to_obs = 1e18
            vehicle_r = 0.0
            
            for (cx, cy, r) in circles:
                dx = cx - ox
                dy = cy - oy
                lx = dx * cos_yaw - dy * sin_yaw
                ly = dx * sin_yaw + dy * cos_yaw
                
                dist_x = max(0.0, abs(lx) - l * 0.5)
                dist_y = max(0.0, abs(ly) - w * 0.5)
                
                dist_to_rect = math.hypot(dist_x, dist_y)
                
                if dist_to_rect < min_dist_to_obs:
                    min_dist_to_obs = dist_to_rect
                    vehicle_r = float(r)

            effective_r = float(vehicle_r + extra_buffer)
            hard_threshold = max(0.0, effective_r - hard_overlap_tol)
            
            if min_dist_to_obs < hard_threshold:
                worst_cost = 1.0
                if allow_hard_collision:
                    hard_collision = True
                    collision_info = (
                        f"ObsRect(x={ox:.1f},y={oy:.1f},l={l:.1f},w={w:.1f}) "
                        f"vs Veh(dist={min_dist_to_obs:.2f},r={effective_r:.2f})"
                    )
                continue

            soft_threshold = effective_r + collision_margin
            if min_dist_to_obs < soft_threshold and collision_margin > 1e-6:
                cost = (soft_threshold - min_dist_to_obs) / collision_margin
                worst_cost = max(worst_cost, float(cost))

    return worst_cost, hard_collision, collision_info


def _check_trajectory_dynamic_collision(
    trajectory, obs, vehicle_length, vehicle_width, start_time, dt_point, config, num_circles=3
):
    """检查轨迹与单个动态障碍物的时空碰撞（多圆包络版本）。"""
    traj_np = np.asarray(trajectory, dtype=float)
    if traj_np.size == 0:
        return 0.0, False

    times = np.asarray(obs.get('time', []), dtype=float)
    xs = np.asarray(obs.get('x', []), dtype=float)
    ys = np.asarray(obs.get('y', []), dtype=float)
    if times.size < 2 or xs.size < 2 or ys.size < 2:
        return 0.0, False

    obs_l = float(obs.get('length', 0.0))
    obs_w = float(obs.get('width', 0.0))
    if obs_l <= 1e-9 or obs_w <= 1e-9:
        return 0.0, False

    yaw_seq = np.asarray(obs.get('yaw', []), dtype=float)
    has_obs_yaw = yaw_seq.size >= 2

    collision_margin = float(config.get('collision_margin', config.get('safety_margin', 0.0)))
    extra_buffer = float(config.get('obstacle_safety_buffer', 0.2))
    extra_buffer = float(max(0.0, extra_buffer))
    hard_overlap_tol = float(config.get('collision_hard_overlap_tolerance', 0.0))
    hard_overlap_tol = float(max(0.0, hard_overlap_tol))

    num_circles = int(max(1, num_circles))
    vehicle_r = 0.5 * float(vehicle_width)
    if num_circles <= 1 or float(vehicle_length) <= 2.0 * vehicle_r + 1e-9:
        offsets = np.asarray([0.0], dtype=float)
    else:
        half = 0.5 * float(vehicle_length) - vehicle_r
        if half <= 1e-9:
            offsets = np.asarray([0.0], dtype=float)
        else:
            offsets = np.linspace(-half, half, num_circles)

    n = int(traj_np.shape[0])
    t_samples = float(start_time) + (np.arange(n, dtype=float) * float(dt_point))
    ox = np.interp(t_samples, times, xs)
    oy = np.interp(t_samples, times, ys)
    if has_obs_yaw:
        oyaw = np.interp(t_samples, times, yaw_seq)
    else:
        oyaw = np.zeros_like(ox)

    worst_cost = 0.0
    hard_collision = False
    effective_r = float(vehicle_r + extra_buffer)
    hard_threshold = max(0.0, effective_r - hard_overlap_tol)
    soft_threshold = effective_r + collision_margin

    for i in range(n):
        px = float(traj_np[i, 0])
        py = float(traj_np[i, 1])
        pyaw = float(traj_np[i, 2]) if traj_np.shape[1] >= 3 else 0.0

        c = float(math.cos(pyaw))
        s = float(math.sin(pyaw))
        obs_cos = float(math.cos(-oyaw[i]))
        obs_sin = float(math.sin(-oyaw[i]))

        min_dist = 1e18
        for off in offsets:
            cx = px + float(off) * c
            cy = py + float(off) * s

            dx = cx - float(ox[i])
            dy = cy - float(oy[i])
            lx = dx * obs_cos - dy * obs_sin
            ly = dx * obs_sin + dy * obs_cos

            dist_x = max(0.0, abs(lx) - 0.5 * obs_l)
            dist_y = max(0.0, abs(ly) - 0.5 * obs_w)
            d = float(math.hypot(dist_x, dist_y))
            if d < min_dist:
                min_dist = d

        if min_dist < hard_threshold:
            hard_collision = True
            worst_cost = 1.0
            break

        if collision_margin > 1e-6 and min_dist < soft_threshold:
            cost = (soft_threshold - min_dist) / max(collision_margin, 1e-6)
            worst_cost = max(worst_cost, float(cost))

    return worst_cost, hard_collision


def check_collision(planner, tentacle, current_time_index):
    """
    检查触须轨迹是否与障碍物碰撞
    """
    config = planner.config
    dt = float(config.get('dt', 0.01))
    planning_horizon = float(config.get('planning_horizon', 5.0))
    start_time = float(current_time_index) * dt

    static_obstacles = planner.environment.get('obstacles', [])
    real_obstacles = [obs for obs in static_obstacles if obs.get('type', 'static') == 'static']

    use_snapshot = bool(config.get('dynamic_snapshot_in_geometric', True))
    snapshot_soft_only = bool(config.get('dynamic_snapshot_soft_only', True))
    dynamic_snapshots = planner._dynamic_obstacles_snapshot(start_time) if use_snapshot else []

    use_spatiotemporal = bool(config.get('dynamic_spatiotemporal_in_geometric', False))
    dynamic_obstacles = getattr(planner, 'dynamic_obstacles', None) or []
    
    total_cost = 0.0
    hard = False
    first_hard_info = ""

    dyn_num_circles = int(config.get('dynamic_collision_num_circles', 3))
    dyn_num_circles = max(1, dyn_num_circles)
    static_num_circles = int(config.get('static_collision_num_circles', 3))
    static_num_circles = max(1, static_num_circles)

    snap_hard_points = int(config.get('dynamic_snapshot_hard_check_points', 1))
    snap_hard_points = max(0, snap_hard_points)

    snap_horizon_s = config.get('dynamic_snapshot_check_horizon_s', None)
    if snap_horizon_s is None:
        snap_horizon_s = float(config.get('replan_interval', 0.1))
    snap_horizon_s = float(max(0.0, snap_horizon_s))

    tractor_trajectory = tentacle.get('tractor_trajectory', tentacle.get('trajectory'))
    if tractor_trajectory is not None and len(tractor_trajectory) > 0:
        if real_obstacles:
            tractor_cost, tractor_hard, info = _check_trajectory_collision(
                tractor_trajectory, real_obstacles, planner.tractor_length, planner.tractor_width,
                config, num_circles=static_num_circles,
            )
            total_cost = max(total_cost, tractor_cost)
            hard = hard or tractor_hard
            if tractor_hard and not first_hard_info: first_hard_info = "Tractor static: " + info

        if dynamic_snapshots:
            prefix_n = int(round(snap_horizon_s / dt)) if dt > 1e-9 else 1
            prefix_n = max(1, prefix_n)
            prefix_n = min(prefix_n, len(tractor_trajectory))

            hard_n = min(prefix_n, max(1, snap_hard_points)) if prefix_n > 0 else 0
            if hard_n > 0:
                head_cost, head_hard, info = _check_trajectory_collision(
                    tractor_trajectory[:hard_n], dynamic_snapshots, planner.tractor_length, planner.tractor_width,
                    config, allow_hard_collision=(not snapshot_soft_only), num_circles=dyn_num_circles,
                )
                total_cost = max(total_cost, head_cost)
                hard = hard or head_hard
                if head_hard and not first_hard_info: first_hard_info = "Tractor dynamic head: " + info

            if prefix_n > hard_n:
                mid_cost, mid_hard, info = _check_trajectory_collision(
                    tractor_trajectory[hard_n:prefix_n], dynamic_snapshots, planner.tractor_length, planner.tractor_width,
                    config, allow_hard_collision=(not snapshot_soft_only), num_circles=dyn_num_circles,
                )
                total_cost = max(total_cost, mid_cost)
                hard = hard or mid_hard
                if mid_hard and not first_hard_info: first_hard_info = "Tractor dynamic mid: " + info

        if use_spatiotemporal and dynamic_obstacles:
            dt_point = planning_horizon / max(len(tractor_trajectory) - 1, 1)
            for obs in dynamic_obstacles:
                dyn_cost, dyn_hard = _check_trajectory_dynamic_collision(
                    trajectory=tractor_trajectory, obs=obs, vehicle_length=planner.tractor_length,
                    vehicle_width=planner.tractor_width, start_time=start_time, dt_point=dt_point,
                    config=config, num_circles=dyn_num_circles,
                )
                total_cost = max(total_cost, dyn_cost)
                hard = hard or dyn_hard
    
    trailer_trajectories = tentacle.get('trailer_trajectories', [])
    if trailer_trajectories is not None and len(trailer_trajectories) > 0:
        for trailer_idx, trailer_traj in enumerate(trailer_trajectories):
            if trailer_traj is None or len(trailer_traj) == 0:
                continue
            trailer_length = planner.trailer_length[trailer_idx]
            trailer_width = planner.trailer_width[trailer_idx]
            
            if real_obstacles:
                trailer_cost, trailer_hard, info = _check_trajectory_collision(
                    trailer_traj, real_obstacles, trailer_length, trailer_width,
                    config, num_circles=static_num_circles,
                )
                total_cost = max(total_cost, trailer_cost)
                hard = hard or trailer_hard
                if trailer_hard and not first_hard_info: first_hard_info = f"Trailer{trailer_idx} static: " + info

            if dynamic_snapshots:
                prefix_n = int(round(snap_horizon_s / dt)) if dt > 1e-9 else 1
                prefix_n = max(1, prefix_n)
                prefix_n = min(prefix_n, len(trailer_traj))

                hard_n = min(prefix_n, max(1, snap_hard_points)) if prefix_n > 0 else 0
                if hard_n > 0:
                    head_cost, head_hard, info = _check_trajectory_collision(
                        trailer_traj[:hard_n], dynamic_snapshots, trailer_length, trailer_width,
                        config, allow_hard_collision=(not snapshot_soft_only), num_circles=dyn_num_circles,
                    )
                    total_cost = max(total_cost, head_cost)
                    hard = hard or head_hard
                    if head_hard and not first_hard_info: first_hard_info = f"Trailer{trailer_idx} dynamic head: " + info

                if prefix_n > hard_n:
                    mid_cost, mid_hard, info = _check_trajectory_collision(
                        trailer_traj[hard_n:prefix_n], dynamic_snapshots, trailer_length, trailer_width,
                        config, allow_hard_collision=(not snapshot_soft_only), num_circles=dyn_num_circles,
                    )
                    total_cost = max(total_cost, mid_cost)
                    hard = hard or mid_hard
                    if mid_hard and not first_hard_info: first_hard_info = f"Trailer{trailer_idx} dynamic mid: " + info

            if use_spatiotemporal and dynamic_obstacles:
                dt_point = planning_horizon / max(len(trailer_traj) - 1, 1)
                for obs in dynamic_obstacles:
                    dyn_cost, dyn_hard = _check_trajectory_dynamic_collision(
                        trajectory=trailer_traj, obs=obs, vehicle_length=trailer_length,
                        vehicle_width=trailer_width, start_time=start_time, dt_point=dt_point,
                        config=config, num_circles=dyn_num_circles,
                    )
                    total_cost = max(total_cost, dyn_cost)
                    hard = hard or dyn_hard
    
    return total_cost, hard, first_hard_info


def check_road_boundary(planner, tentacle):
    """
    检查触须轨迹是否越出道路边界
    """
    config = planner.config
    environment = planner.environment
    left_boundary = environment['left_boundary']
    right_boundary = environment['right_boundary']
    
    has_left = left_boundary is not None and len(left_boundary) > 0
    has_right = right_boundary is not None and len(right_boundary) > 0
    
    if not has_left and not has_right:
        return 0.0, False
    
    boundary_margin = float(config.get('boundary_margin', 0.0))

    def eval_corridor_cost(traj, vehicle_length, vehicle_width):
        num_circles = 3
        vehicle_radius = float(vehicle_width) / 2.0

        def _get_circles(px, py, yaw):
            r = vehicle_radius
            if num_circles <= 1 or float(vehicle_length) <= 2.0 * r + 1e-9:
                return [(float(px), float(py), float(r))]
            half = 0.5 * float(vehicle_length) - r
            if half <= 1e-9:
                return [(float(px), float(py), float(r))]
            offs = np.linspace(-half, half, num_circles)
            cy = float(math.sin(float(yaw)))
            cx = float(math.cos(float(yaw)))
            out = []
            for off in offs:
                out.append((float(px) + float(off) * cx, float(py) + float(off) * cy, float(r)))
            return out

        worst = 0.0
        hard_v = False
        debug_info = ""

        all_boundaries = []
        if has_left:
            all_boundaries.append(left_boundary)
        if has_right:
            all_boundaries.append(right_boundary)
        
        if not all_boundaries:
            return 0.0, False

        for p in traj:
            px, py = float(p[0]), float(p[1])
            yaw = float(p[2]) if len(p) >= 3 else 0.0

            circles = _get_circles(px, py, yaw)
            for (cx, cy, cr) in circles:
                for boundary in all_boundaries:
                    dists_sq = (boundary[:, 0] - cx) ** 2 + (boundary[:, 1] - cy) ** 2
                    min_idx = int(np.argmin(dists_sq))
                    d_min = float(math.sqrt(float(dists_sq[min_idx])))

                    hard_tol = float(config.get('boundary_hard_tolerance', 0.05))
                    hard_tol = float(max(0.0, hard_tol))
                    if d_min < cr - hard_tol:
                        hard_v = True
                        if not debug_info:
                            debug_info = (
                                f"hard: circle=({cx:.2f},{cy:.2f},r={cr:.2f}) "
                                f"d_min={d_min:.3f} idx={min_idx}"
                            )
                        worst = 1.0
                        break

                    if boundary_margin > 1e-9:
                        soft_th = cr + boundary_margin
                        if d_min < soft_th:
                            soft = (soft_th - d_min) / max(boundary_margin, 1e-9)
                            worst = max(worst, float(max(0.0, min(1.0, soft))))

                if hard_v:
                    break
            
            if hard_v:
                break

        return worst, hard_v, debug_info

    tractor_traj = tentacle.get('tractor_trajectory', tentacle.get('trajectory'))
    if tractor_traj is None or len(tractor_traj) == 0:
        return 0.0, False

    worst_cost, hard_violation, dbg = eval_corridor_cost(tractor_traj, planner.tractor_length, planner.tractor_width)
    if dbg:
        tentacle['boundary_debug'] = f"tractor {dbg}"

    trailer_trajectories = tentacle.get('trailer_trajectories', [])
    if trailer_trajectories is not None and len(trailer_trajectories) > 0:
        for trailer_idx, trailer_traj in enumerate(trailer_trajectories):
            if trailer_traj is None or len(trailer_traj) == 0:
                continue
            trailer_length = planner.trailer_length[trailer_idx]
            trailer_width = planner.trailer_width[trailer_idx]
            c, h, dbg = eval_corridor_cost(trailer_traj, trailer_length, trailer_width)
            worst_cost = max(worst_cost, c)
            hard_violation = hard_violation or h
            if dbg:
                if (not tentacle.get('boundary_debug')) or h:
                    tentacle['boundary_debug'] = f"trailer{trailer_idx} {dbg}"

    return worst_cost, hard_violation


def compute_smoothness_cost(tentacle):
    """
    计算触须轨迹的平滑度代价
    """
    trajectory = tentacle.get('tractor_trajectory', tentacle.get('trajectory'))
    if len(trajectory) < 3:
        return 0.0
    
    curvature_sum = 0.0
    
    for i in range(1, len(trajectory) - 1):
        p1 = np.array(trajectory[i-1][:2])
        p2 = np.array(trajectory[i][:2])
        p3 = np.array(trajectory[i+1][:2])
        
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)
        
        if a > 0 and b > 0 and c > 0:
            s = (a + b + c) / 2
            area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
            curvature = 4 * area / (a * b * c) if (a * b * c) > 0 else 0
            curvature_sum += curvature
    
    return curvature_sum


def evaluate_tentacle_cost(planner, tentacle, current_time_index):
    """
    评估触须轨迹的代价函数
    """
    config = planner.config
    collision_cost, hard_collision, col_info = check_collision(planner, tentacle, current_time_index)
    boundary_cost, hard_boundary = check_road_boundary(planner, tentacle)

    traj = tentacle.get('tractor_trajectory', [])
    target_offset = 0.0
    if traj and len(traj) > 5:
        start_yaw = traj[0][2]
        end_yaw = traj[-1][2]
        dyaw = end_yaw - start_yaw
        dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
        
        if abs(dyaw) > 0.2:
            swing_direction = -1.0 if dyaw > 0 else 1.0
            target_offset = swing_direction * 1.5
            tentacle['swing_active'] = True

    lateral_base = abs(tentacle['lateral_offset'] - target_offset) * 2.0
    safe_gain = float(config.get('lateral_safe_gain', 0.0))
    risk = float(max(collision_cost, boundary_cost))
    safe_factor = float(max(0.0, min(1.0, 1.0 - risk)))
    lateral_cost = lateral_base * (1.0 + safe_gain * safe_factor)

    centerline_cost = 0.0
    centerline = None
    if planner.environment:
        centerline = planner.environment.get('centerline', None)
    if centerline is not None and len(centerline) > 1:
        traj = tentacle.get('tractor_trajectory', [])
        if traj:
            path_xy = np.array(centerline, dtype=float)
            path_s = _compute_arc_length_xy(path_xy)
            stride = int(max(1, config.get('centerline_cost_stride', 3)))
            dists = []
            for i in range(0, len(traj), stride):
                _, _, dist = _project_xy_to_path_s_idx_dist(path_xy, path_s, traj[i][0], traj[i][1])
                dists.append(dist)
            if dists:
                centerline_cost = float(np.mean(dists))

    smoothness_cost = compute_smoothness_cost(tentacle)
    
    lateral_w = float(config.get('lateral_cost_weight', 10.0))
    centerline_w = float(config.get('centerline_cost_weight', 0.0))
    total_cost = (
        collision_cost * 1000.0 +
        boundary_cost * 500.0 +
        lateral_cost * lateral_w +
        centerline_cost * centerline_w +
        smoothness_cost * 0.5
    )
    
    if hard_collision:
        tentacle['is_valid'] = False
        tentacle['reject_reason'] = f"hard_collision {col_info} (cost={collision_cost:.2f})"
    elif hard_boundary:
        tentacle['is_valid'] = False
        dbg = tentacle.get('boundary_debug', '')
        tail = f" | {dbg}" if dbg else ""
        tentacle['reject_reason'] = f"hard_boundary (cost={boundary_cost:.2f}){tail}"
    
    return total_cost


def run_single_replan(
    planner,
    current_pos: Sequence[float],
    current_yaw: float,
    current_speed: float,
    current_trailer_states: Optional[List[Dict[str, Any]]] = None,
    elapsed_time: float = 0.0,
    reference_segment: Optional[Sequence[Sequence[float]]] = None,
    lateral_offsets: Optional[Sequence[OffsetSpec]] = None,
    include_all_tentacles: bool = True,
) -> SingleReplanResult:
    """
    执行一次重规划，输出本次最优触须轨迹和速度。

    Args:
        planner: TentacleLocalPlanner 实例。
        current_pos: 当前牵引车位置 [x, y]。
        current_yaw: 当前牵引车朝向（rad）。
        current_speed: 当前牵引车速度 (m/s)。
        current_trailer_states: 当前挂车状态；若为空则按当前牵引车状态初始化。
        elapsed_time: 当前累计仿真时间（秒），用于动态障碍物时序对齐。
        reference_segment: 可选。若不传，则从 planner.global_trajectory 自动截取。
        lateral_offsets: 可选。若传入，则只跑这一组偏移，不再执行三阶段 fallback。
        include_all_tentacles: 返回中是否包含全部候选触须。

    Returns:
        SingleReplanResult
    """
    if planner.environment is None:
        return SingleReplanResult(
            success=False,
            best_tentacle=None,
            speed_profile=None,
            tentacles=[],
            valid_count=0,
            total_count=0,
            phase="none",
            current_time_index=0,
            current_step=None,
            reference_segment=[],
            timing={},
            reason="planner.environment is None; please call load_environment first",
        )

    current_pos_np = np.asarray(current_pos, dtype=float).reshape(-1)
    if current_pos_np.size != 2:
        raise ValueError(f"current_pos must be length-2, got shape={current_pos_np.shape}")

    if current_trailer_states is None:
        tractor_state = {
            "x": float(current_pos_np[0]),
            "y": float(current_pos_np[1]),
            "yaw": float(current_yaw),
        }
        current_trailer_states = planner.kinematics.initialize_trailers(tractor_state)

    current_step: Optional[int] = None
    if reference_segment is None:
        try:
            reference_segment, current_step = _build_reference_segment(planner, current_pos_np)
        except ValueError as exc:
            return SingleReplanResult(
                success=False,
                best_tentacle=None,
                speed_profile=None,
                tentacles=[],
                valid_count=0,
                total_count=0,
                phase="none",
                current_time_index=0,
                current_step=None,
                reference_segment=[],
                timing={},
                reason=str(exc),
            )

    ref_segment_list = [list(map(float, p)) for p in reference_segment]
    if len(ref_segment_list) < 2:
        return SingleReplanResult(
            success=False,
            best_tentacle=None,
            speed_profile=None,
            tentacles=[],
            valid_count=0,
            total_count=0,
            phase="none",
            current_time_index=0,
            current_step=current_step,
            reference_segment=ref_segment_list,
            timing={},
            reason="reference_segment has less than 2 points",
        )

    current_time_index = int(round(float(elapsed_time) / float(planner.config["dt"])))
    timing: Dict[str, float] = {}
    max_lat_dev = float(planner.config.get("lateral_deviation", 3.5))
    num_primary = int(max(1, planner.config.get("num_tentacles", 1)))
    primary_offsets = list(np.linspace(-max_lat_dev, max_lat_dev, num_primary))

    if lateral_offsets is not None:
        generate_start = time.time()
        tentacles, valid_count = _generate_and_eval(
            planner=planner,
            current_pos=current_pos_np,
            current_yaw=float(current_yaw),
            current_trailer_states=current_trailer_states,
            reference_segment=ref_segment_list,
            lateral_offsets=lateral_offsets,
            current_time_index=current_time_index,
        )
        timing["generate_and_eval"] = float(time.time() - generate_start)

        if not tentacles:
            return SingleReplanResult(
                success=False,
                best_tentacle=None,
                speed_profile=None,
                tentacles=[],
                valid_count=0,
                total_count=0,
                phase="custom",
                current_time_index=current_time_index,
                current_step=current_step,
                reference_segment=ref_segment_list,
                timing=timing,
                reason="no tentacles generated from custom lateral_offsets",
            )

        if valid_count > 0:
            valid_tentacles = [t for t in tentacles if t.get("is_valid")]
            best_tentacle = min(valid_tentacles, key=lambda t: t["cost"])
        else:
            best_tentacle = min(tentacles, key=lambda t: t["cost"])

        speed_plan_start = time.time()
        speed_profile, combined_trajectory = build_speed_and_trajectory(
            planner, best_tentacle, float(current_speed), elapsed_time
        )
        timing['speed_plan'] = time.time() - speed_plan_start

        return SingleReplanResult(
            success=True,
            best_tentacle=best_tentacle,
            speed_profile=speed_profile,
            trajectory=combined_trajectory,
            tentacles=tentacles if include_all_tentacles else [],
            valid_count=valid_count,
            total_count=len(tentacles),
            phase="custom",
            current_time_index=current_time_index,
            current_step=current_step,
            reference_segment=ref_segment_list,
            timing=timing,
            reason="",
        )

    # 默认流程：复用原主代码的三阶段搜索
    # phase1: 中心 primary_n 条
    # phase2: 15 条二维偏移 fallback
    # phase3: 扩大横向范围后的二维偏移 fallback
    primary_n = int(max(1, planner.config.get('primary_tentacles', 5)))
    fallback_n = int(max(1, planner.config.get('fallback_tentacles', 15)))
    primary_span_factor = float(planner.config.get('primary_tentacle_span_factor', 2.0 / 7.0))

    max_lateral = float(planner.config.get('lateral_deviation', 3.5))
    wide_lateral = float(planner.config.get('lateral_deviation_fallback_wide', 5.0))
    wide_lateral = float(max(max_lateral, wide_lateral))

    primary_span = float(max(0.0, max_lateral * primary_span_factor))
    primary_offsets = list(np.linspace(-primary_span, primary_span, primary_n))
    fallback_offsets = _build_fallback_offsets_2d(primary_offsets, fallback_n, max_lateral)
    fallback_offsets_wide = _build_fallback_offsets_2d(primary_offsets, fallback_n, wide_lateral)

    phase_used = 'phase1'

    phase1_start = time.time()
    tentacles, valid_count = _generate_and_eval(
        planner=planner,
        current_pos=current_pos_np,
        current_yaw=float(current_yaw),
        current_trailer_states=current_trailer_states,
        reference_segment=ref_segment_list,
        lateral_offsets=primary_offsets,
        current_time_index=current_time_index,
    )
    timing['phase1_gen_eval'] = time.time() - phase1_start

    if not tentacles:
        return SingleReplanResult(
            success=False,
            best_tentacle=None,
            speed_profile=None,
            trajectory=None,
            tentacles=[],
            valid_count=0,
            total_count=0,
            phase='phase1',
            current_time_index=current_time_index,
            current_step=current_step,
            reference_segment=ref_segment_list,
            timing=timing,
            reason='no tentacles generated in phase1',
        )

    if valid_count == 0 and fallback_offsets:
        phase2_start = time.time()
        phase2_tentacles, phase2_valid = _generate_and_eval(
            planner=planner,
            current_pos=current_pos_np,
            current_yaw=float(current_yaw),
            current_trailer_states=current_trailer_states,
            reference_segment=ref_segment_list,
            lateral_offsets=fallback_offsets,
            current_time_index=current_time_index,
        )
        timing['phase2_gen_eval'] = time.time() - phase2_start
        if phase2_tentacles:
            tentacles = phase2_tentacles
            valid_count = phase2_valid
            phase_used = 'phase2_fallback'

    if valid_count == 0 and wide_lateral > max_lateral + 1e-6 and fallback_offsets_wide:
        phase3_start = time.time()
        phase3_tentacles, phase3_valid = _generate_and_eval(
            planner=planner,
            current_pos=current_pos_np,
            current_yaw=float(current_yaw),
            current_trailer_states=current_trailer_states,
            reference_segment=ref_segment_list,
            lateral_offsets=fallback_offsets_wide,
            current_time_index=current_time_index,
        )
        timing['phase3_gen_eval'] = time.time() - phase3_start
        if phase3_tentacles:
            tentacles = phase3_tentacles
            valid_count = phase3_valid
            phase_used = 'phase3_fallback'

    if not tentacles:
        return SingleReplanResult(
            success=False,
            best_tentacle=None,
            speed_profile=None,
            trajectory=None,
            tentacles=[],
            valid_count=0,
            total_count=0,
            phase=phase_used,
            current_time_index=current_time_index,
            current_step=current_step,
            reference_segment=ref_segment_list,
            timing=timing,
            reason='no tentacles generated after fallback',
        )

    best_tentacle = None
    if valid_count > 0:
        valid_tentacles = [t for t in tentacles if t.get('is_valid')]
        best_tentacle = min(valid_tentacles, key=lambda t: t['cost'])
    elif tentacles:
        # 与原流程一致：无有效触须时选择代价最小的无效触须尝试执行
        best_tentacle = min(tentacles, key=lambda t: t['cost'])

    if best_tentacle is None:
        return SingleReplanResult(
            success=False,
            best_tentacle=None,
            speed_profile=None,
            trajectory=None,
            tentacles=tentacles if include_all_tentacles else [],
            valid_count=valid_count,
            total_count=len(tentacles),
            phase=phase_used,
            current_time_index=current_time_index,
            current_step=current_step,
            reference_segment=ref_segment_list,
            timing=timing,
            reason='failed to select best tentacle',
        )

    timing['select'] = 0.0

    speed_plan_start = time.time()
    speed_profile, combined_trajectory = build_speed_and_trajectory(
        planner, best_tentacle, float(current_speed), elapsed_time
    )
    timing['speed_plan'] = time.time() - speed_plan_start

    return SingleReplanResult(
        success=True,
        best_tentacle=best_tentacle,
        speed_profile=speed_profile,
        trajectory=combined_trajectory,
        tentacles=tentacles if include_all_tentacles else [],
        valid_count=valid_count,
        total_count=len(tentacles),
        phase=phase_used,
        current_time_index=current_time_index,
        current_step=current_step,
        reference_segment=ref_segment_list,
        timing=timing,
    )
