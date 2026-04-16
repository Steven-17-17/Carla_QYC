"""
触须法局部路径规划器 (Tentacle-based Local Path Planner)

功能：
1. 读取全局路径规划结果（trajectory.txt）
2. 读取道路环境信息（environment.txt）- 道路边界和静态障碍物
3. 基于触须法生成局部避障轨迹
4. 可视化展示局部规划过程

作者：黄梓谦、蒋涛
日期：2025-1-27
版本：v2.1 - 简化版（以移除不需要的库和可视化，仅输出局部路径）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation, PillowWriter
import csv
import math
from datetime import datetime
import os
import time
import sys
from multi_trailer_kinematics_final import MultiTrailerKinematics, VehicleParams
from single_replan_interface import (
    run_single_replan,
    evaluate_tentacle_cost,
    check_collision,
    check_road_boundary,
    build_speed_and_trajectory,
)


class TentacleLocalPlanner:
    """触须法局部路径规划器"""
    
    def __init__(self, config=None, num_trailers=1):
        """
        初始化局部路径规划器
        
        Args:
            config: 配置参数字典（可选）
            num_trailers: 挂车节数（默认为1，支持1-8节）
        """
        # 默认配置
        self.config = {
            'planning_horizon': 6.0,      # 规划时域 (s)
            'replan_interval': 1.0,       # 重规划间隔 (s) - 缩短重规划间隔
            'dt': 0.01,                   # 时间步长 (s) - 与全局规划器一致（10ms）
            'num_tentacles': 7,          # 触须数量
            'lateral_deviation': 3.5,     # 最大横向偏移 (m)
            # 当 15 条触须全部失效时，进一步扩大横向搜索范围（单位：m）
            # 例如从 ±3.5m 扩展到 ±5.0m 再尝试一次。
            'lateral_deviation_fallback_wide': 5.0,
            # 触须“几何长度”（沿参考段的距离）。用于将触须长度固定为约 15m，
            # 避免 planning_horizon 被外部配置成 5s 时，触须过长导致边界/障碍物过度触发。
            'tentacle_length_m': 15.0,

            # 执行/重规划策略：每次生成一条局部路径后，默认执行该路径的比例再重规划。
            # 例如 0.5 表示“走到 1/2 再重规划”，避免像 30Hz 重规划那样每次只走 0.2m。
            # 取值范围建议 (0,1]；若 <=0 则退化回旧策略（按 replan_interval 执行）。
            'execute_progress_ratio': 0.5,
            'no_valid_stop_threshold': 8, # 放宽连续无效触须阈值，避免短时抖动就触发原地刹停
            'no_valid_stop_duration_s': 0.8, # 若确实必须停，缩短停等时间，减少 v-t 图上的长零速平台
            'hysteresis_switch_margin': 50.0, # 新路径需明显更优才切换，抑制左右横跳
            'hysteresis_low_speed_switch_margin': 5.0, # 低速爬行时显著降低保持旧路径的惯性
            'hysteresis_min_speed_for_obstacle_hold': 2.0, # 速度掉低后不再为了“保持原道路”强行锁住旧路径
            'hysteresis_max_obs_cost_for_hold': 0.08, # 仅当障碍物冲突很轻微时才忽略旧路径的碰撞代价
            'safety_margin': 2.0,         # 安全余量 (m) - 增加安全距离
            'collision_margin': 1.05,      # 碰撞检测余量 (m) - 单圆模型使用，避免过度保守
            # 障碍物碰撞额外安全裕度 (m)：在车辆外包络半径上再加一圈 buffer。
            # 例如 0.2m 表示与障碍物保持至少 20cm 的额外间隙。
            'obstacle_safety_buffer': 0.2,
            'boundary_margin': 0.2,       # 边界余量 (m) - 距离边界过近的软惩罚
            # 硬越界容差（允许的“重叠”）：默认 10cm。
            # 说明：hard 判据是 d_min < r - tol。
            # 你的日志里 trailer7 的 d_min≈0.93~0.99, r=1.0（仅几厘米重叠/贴边），若 tol=0 会全部判死。
            'boundary_hard_tolerance': 0.0,

            # 纵向（速度）规划参数
            'v_max': 6.94,                # 最高速度 (m/s) ≈ 25km/h
            'a_max': 0.6,                 # 最大加速度 (m/s^2)
            'a_min': -2.0,                # 最大减速度 (m/s^2)（常规制动）
            'a_lat_max': 4.5,             # 放宽曲率限速，避免跟车后段长期被横向限速压在明显低于前车的速度
            'initial_speed': 2.0,         # 初始速度 (m/s)
            'follow_headway': 0.30,      # 跟驰时间头距（s）
            'follow_min_gap': 0.20,      # 最小跟驰间距（m）
            'follow_gain': 0.55,         # 跟驰收敛增益
            'w_progress': 1.0,           # 前进奖励权重
            'w_speed': 1.80,             # 速度跟踪权重
            'w_acc': 0.05,               # 加速度权重
            'w_smooth': 1.60,            # 平滑/jerk 权重
            'w_safety': 10.0,            # 安全间距惩罚权重

            # 动态障碍物处理策略
            # - 几何(横向)规划：把“当前时刻”的动态障碍物快照当成静态障碍物去绕行（软代价，不判死）
            # - 速度规划：继续使用未来轨迹做时空避障
            'dynamic_snapshot_in_geometric': True,
            'dynamic_snapshot_soft_only': True,
            # 可选：若你确实需要几何层也做时空碰撞（非常保守，容易无有效触须），才打开
            'dynamic_spatiotemporal_in_geometric': False,

            # 动态障碍物的几何碰撞：使用多圆模型降低“包围大圆”带来的过度保守
            'dynamic_collision_num_circles': 3,
            # 即使动态快照总体采用软代价，也不允许“起始就重叠”（避免 cost~1000 仍执行）
            'dynamic_snapshot_hard_check_points': 1,

            # 动态快照只约束“短时段”（避免把未来 5s 都当成当前障碍物占据，从而长期贴边）
            # None 表示使用 replan_interval；也可以显式给秒数
            'dynamic_snapshot_check_horizon_s': None,

            # 回到参考线倾向：越安全（无碰撞/无越界风险）越强地惩罚横向偏移
            # 将中心线吸引代价降到最低：默认不惩罚横向偏移（更愿意借道/摆出以让多挂车通过）
            'lateral_safe_gain': 0.0,
            'lateral_cost_weight': 3.0,
            # 中心线吸引代价（按与中心线的平均距离）
            'centerline_cost_weight': 2.0,
            'centerline_cost_stride': 3,
            'terminal_path_extension_enable': True,  # 终点附近允许按末端切向继续外推参考路径，避免跟车场景过早收油
            'terminal_path_extension_m': 25.0,       # 末端参考路径外推长度（m）
            'terminal_path_stop_remaining_steps': 10, # 仅当剩余全局轨迹极短时才真正停止规划
        }
        
        # 更新配置
        if config:
            self.config.update(config)

        # 允许用频率配置重规划（更贴近工程里的控制/重规划频率）
        # 优先级：replan_hz > control_hz > replan_interval
        replan_hz = self.config.get('replan_hz', None)
        if replan_hz is None:
            replan_hz = self.config.get('control_hz', None)
        if replan_hz is not None:
            replan_hz = float(replan_hz)
            if replan_hz <= 0:
                raise ValueError(f"replan_hz/control_hz must be > 0, got {replan_hz}")
            self.config['replan_hz'] = replan_hz
            self.config['replan_interval'] = 1.0 / replan_hz
        else:
            # 反向补齐，便于打印和动画默认值一致
            self.config['replan_hz'] = 1.0 / float(self.config['replan_interval'])
        
        # 数据存储
        self.global_trajectory = None
        self.environment = None
        self.dynamic_obstacles = []  # 动态障碍物（未来轨迹点序列）
        
        # 车辆参数
        self.tractor_length = 3.3        # 牵引车长度
        self.tractor_width = 2.0         # 牵引车宽度
        
        # 挂车配置
        self.num_trailers = num_trailers
        # 这里的 [1.4, 2.65] 分别是挂钩到前轴/后轴的长度参数吗？
        # 我们当前环境里的挂车非常大（甚至跟集装箱差不多），为了测试通过我们先把包络适当加大
        self.trailer_length = [5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5][:num_trailers]
        self.trailer_width = [2.5] * num_trailers
        
        # 初始化运动学模型
        vehicle_params = VehicleParams(num_trailers=num_trailers)
        self.kinematics = MultiTrailerKinematics(vehicle_params, dt=self.config['dt'])
        
        # 挂车状态
        self.trailer_states = None
        self.last_exec_acc = 0.0
        self.perf_stats = {
            'replan_count': 0,
            'total_replan_time': 0.0,
            'valid_tentacles': 0,
            'total_tentacles': 0,
            'phase_counts': {
                'phase1': 0,
                'phase2_fallback': 0,
                'phase3_fallback': 0,
                'custom': 0,
            },
        }
        
        print("=" * 70)
        print("触须法局部路径规划器 v2.0")
        print("=" * 70)
        print(f"配置参数:")
        print(f"  规划时域: {self.config['planning_horizon']}s")
        print(f"  重规划间隔: {self.config['replan_interval']}s (~{self.config['replan_hz']:.1f}Hz)")
        print(f"  时间步长: {self.config['dt']}s")
        print(f"  触须数量: {self.config['num_tentacles']}条")
        print(f"  横向偏移范围: ±{self.config['lateral_deviation']}m")
        print(f"  安全余量: {self.config['safety_margin']}m")
        print(f"  车辆配置: 牵引车 + {self.num_trailers}节挂车")

        print(f"  纵向约束: v_max={self.config['v_max']:.2f}m/s, a_max={self.config['a_max']:.2f}m/s^2, a_min={self.config['a_min']:.2f}m/s^2")
        print(
            f"  跟驰核心: headway={self.config['follow_headway']:.2f}s, "
            f"min_gap={self.config['follow_min_gap']:.2f}m, gain={self.config['follow_gain']:.2f}"
        )
        print(
            f"  代价权重: progress={self.config['w_progress']:.2f}, speed={self.config['w_speed']:.2f}, "
            f"acc={self.config['w_acc']:.2f}, smooth={self.config['w_smooth']:.2f}, safety={self.config['w_safety']:.2f}"
        )

    # =========================
    #  纵向（速度）规划工具函数
    # =========================
    @staticmethod
    def _angle_lerp(a0, a1, t):
        da = math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))
        return a0 + t * da

    @staticmethod
    def _compute_arc_length_xy(xy):
        if xy is None or len(xy) == 0:
            return np.array([], dtype=float)
        s = np.zeros(len(xy), dtype=float)
        for i in range(1, len(xy)):
            dx = float(xy[i, 0] - xy[i - 1, 0])
            dy = float(xy[i, 1] - xy[i - 1, 1])
            s[i] = s[i - 1] + float(math.hypot(dx, dy))
        return s

    @staticmethod
    def _find_segment_by_s(s_points, s_query):
        """返回 (i0, i1, t) 使得 s_query 在 [s[i0], s[i1]] 内，t∈[0,1]。"""
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

    def _interp_traj_by_index(self, traj, i0, i1, t):
        """按离散索引 (i0,i1,t) 在 traj 上插值，保持与牵引车同一“进度参数”。"""
        i0 = int(max(0, min(i0, len(traj) - 1)))
        i1 = int(max(0, min(i1, len(traj) - 1)))
        t = float(max(0.0, min(1.0, t)))
        p0 = traj[i0]
        p1 = traj[i1]
        x = (1.0 - t) * float(p0[0]) + t * float(p1[0])
        y = (1.0 - t) * float(p0[1]) + t * float(p1[1])
        if len(p0) >= 3 and len(p1) >= 3:
            yaw = self._angle_lerp(float(p0[2]), float(p1[2]), t)
        else:
            dx = float(p1[0] - p0[0])
            dy = float(p1[1] - p0[1])
            yaw = float(math.atan2(dy, dx)) if (abs(dx) + abs(dy)) > 1e-12 else 0.0
        return np.array([x, y, yaw], dtype=float)

    def _project_xy_to_path_s_idx_dist(self, path_xy, path_s, x, y):
        """返回 (s, idx, dist)；用于纵向检测里的“横向门控”。

        与“最近离散点”相比，连续投影能更准确反映障碍物沿路径的前后关系，
        避免弯道中把前方同向车辆的 s 低估，从而错误压低跟驰速度。
        """
        if path_xy is None or len(path_xy) == 0:
            return 0.0, 0, 1e18

        n = int(len(path_xy))
        qx = float(x)
        qy = float(y)
        if n == 1:
            dx = float(path_xy[0, 0]) - qx
            dy = float(path_xy[0, 1]) - qy
            return 0.0, 0, float(math.hypot(dx, dy))

        best_d2 = float('inf')
        best_s = 0.0
        best_idx = 0
        for k in range(n - 1):
            x0 = float(path_xy[k, 0])
            y0 = float(path_xy[k, 1])
            x1 = float(path_xy[k + 1, 0])
            y1 = float(path_xy[k + 1, 1])
            ux = x1 - x0
            uy = y1 - y0
            seg_len2 = ux * ux + uy * uy
            if seg_len2 <= 1e-12:
                t = 0.0
            else:
                t = ((qx - x0) * ux + (qy - y0) * uy) / seg_len2
                t = max(0.0, min(1.0, t))

            px = x0 + t * ux
            py = y0 + t * uy
            dx = qx - px
            dy = qy - py
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                seg_len = math.sqrt(seg_len2) if seg_len2 > 1e-12 else 0.0
                base_s = float(path_s[k]) if path_s is not None and len(path_s) > k else 0.0
                best_s = base_s + t * seg_len
                best_idx = k if t < 0.5 else min(k + 1, n - 1)

        return float(best_s), int(best_idx), float(math.sqrt(best_d2))

    @staticmethod
    def _path_heading_at_idx(path_xy, idx):
        """估计离散路径在 idx 附近的切向航向。"""
        n = 0 if path_xy is None else len(path_xy)
        if n <= 1:
            return 0.0
        idx = int(max(0, min(idx, n - 1)))
        i0 = max(0, idx - 1)
        i1 = min(n - 1, idx + 1)
        if i1 == i0:
            if idx <= 0:
                i1 = min(n - 1, 1)
            else:
                i0 = max(0, n - 2)
        dx = float(path_xy[i1, 0] - path_xy[i0, 0])
        dy = float(path_xy[i1, 1] - path_xy[i0, 1])
        return float(math.atan2(dy, dx)) if (abs(dx) + abs(dy)) > 1e-12 else 0.0

    def _dynamic_obstacle_state_at_time(self, obs, t):
        """线性插值获取动态障碍物在时间 t 的状态。"""
        times = obs['time']
        if t <= times[0]:
            i0 = 0
            i1 = 0
            alpha = 0.0
        elif t >= times[-1]:
            i0 = len(times) - 1
            i1 = i0
            alpha = 0.0
        else:
            i1 = int(np.searchsorted(times, t, side='right'))
            i1 = max(1, min(i1, len(times) - 1))
            i0 = i1 - 1
            dt = float(times[i1] - times[i0])
            alpha = 0.0 if dt <= 1e-9 else float((t - times[i0]) / dt)
            alpha = max(0.0, min(1.0, alpha))

        x = (1.0 - alpha) * float(obs['x'][i0]) + alpha * float(obs['x'][i1])
        y = (1.0 - alpha) * float(obs['y'][i0]) + alpha * float(obs['y'][i1])
        yaw = self._angle_lerp(float(obs['yaw'][i0]), float(obs['yaw'][i1]), alpha)
        v = (1.0 - alpha) * float(obs['velocity'][i0]) + alpha * float(obs['velocity'][i1])
        return x, y, yaw, v

    def _dynamic_obstacles_snapshot(self, t):
        """把动态障碍物在时刻 t 的位置快照成“静态障碍物”格式。"""
        snapshots = []
        dyn = getattr(self, 'dynamic_obstacles', None) or []
        for obs in dyn:
            try:
                x, y, yaw, _v = self._dynamic_obstacle_state_at_time(obs, float(t))
                snapshots.append({
                    'x': float(x),
                    'y': float(y),
                    'yaw': float(yaw),
                    'length': float(obs.get('length', 0.0)),
                    'width': float(obs.get('width', 0.0)),
                    'type': 'dynamic_snapshot'
                })
            except Exception:
                continue
        return snapshots

    def _draw_single_vehicle(self, ax, state, length, width, color, label=''):
        """
        绘制单个车辆（根据真实尺寸）
        
        Args:
            ax: matplotlib轴对象
            state: 状态字典 {'x': ..., 'y': ..., 'yaw': ...}
            length: 车辆长度
            width: 车辆宽度
            color: 颜色
            label: 标签
        """
        x, y, yaw = state['x'], state['y'], state['yaw']
        
        # 旋转矩阵
        c = math.cos(yaw)
        s = math.sin(yaw)
        hf = 2.65
        hr = 0.65
        hw = 0.5 * width
        
        # 车辆四个角的坐标（以后轴中心为原点）
        p1 = np.array([x + hf * c - hw * s, y + hf * s + hw * c])
        p2 = np.array([x - hr * c - hw * s, y - hr * s + hw * c])
        p3 = np.array([x - hr * c + hw * s, y - hr * s - hw * c])
        p4 = np.array([x + hf * c + hw * s, y + hf * s - hw * c])
        
        # 绘制车辆
        points = np.array([p1, p2, p3, p4, p1])
        line_obj = ax.plot(points[:, 0], points[:, 1], color=color, linewidth=0.8, label=label)[0]
        line_obj.set_gid('vehicle_current')  # 标记为当前车辆
        
        # 绘制填充
        poly = ax.fill(points[:-1, 0], points[:-1, 1], color=color, alpha=0.3)[0]
        poly.set_gid('vehicle_current')  # 标记为当前车辆
        
        # 绘制方向箭头
        arrow_len = length / 4
        arrow_end = np.array([x + arrow_len * c, y + arrow_len * s])
        ax.arrow(x, y, arrow_end[0] - x, arrow_end[1] - y,
            head_width=0.3, head_length=0.2, fc=color, ec=color, linewidth=0.6,
                gid='vehicle_current')  # 标记为当前车辆
    
    def _draw_vehicle_convoy(self, ax, tractor_state, color_scheme=None, label_prefix='', show_label=True):
        """
        绘制牵引车
        
        Args:
            ax: matplotlib轴对象
            tractor_state: 牵引车状态 {'x': ..., 'y': ..., 'yaw': ...}
            color_scheme: 颜色方案（可选）
            label_prefix: 标签前缀
            show_label: 是否显示标签
        """
        if color_scheme is None:
            color_scheme = ['#2E7D32']
        
        # 绘制牵引车
        tractor_color = color_scheme[0]
        tractor_label = f'{label_prefix}Tractor' if show_label else ''
        self._draw_single_vehicle(ax, tractor_state, self.tractor_length, 
                                 self.tractor_width, tractor_color, tractor_label)
    
    def _get_vehicle_state_from_trajectory(self, trajectory, step_idx):
        """
        从轨迹数据中提取指定步骤的车辆状态
        
        Args:
            trajectory: 轨迹数据字典
            step_idx: 步骤索引
            
        Returns:
            tuple: (tractor_state, trailer_states)
        """
        tractor_state = {
            'x': trajectory['tractor_x'][step_idx],
            'y': trajectory['tractor_y'][step_idx],
            'yaw': trajectory['tractor_yaw'][step_idx]
        }
        
        return tractor_state
    
    def load_global_trajectory(self, filename):
        """
        读取全局路径规划结果
        
        Args:
            filename: 轨迹文件路径
            
        Returns:
            bool: 读取是否成功
        """
        print(f"\n[1/3] 读取全局轨迹...")
        print(f"    文件: {os.path.basename(filename)}")
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 解析文件头
            header = lines[0].strip().split(',')
            
            # 初始化数据字典
            data = {}
            for col in header:
                data[col] = []
            
            # 读取数据行
            for line in lines[1:]:
                values = line.strip().split(',')
                for i, col in enumerate(header):
                    data[col].append(float(values[i]))
            
            self.global_trajectory = data
            
            # 统计信息
            num_steps = len(data['step'])
            
            print(f"    [OK] 读取成功")
            print(f"      - 轨迹点数: {num_steps}")
            print(f"      - 牵引车: 1辆")
            
            return True
            
        except Exception as e:
            print(f"    [ERR] 读取失败: {e}")
            return False
    
    def load_environment(self, filename):
        """
        读取环境信息（道路边界和静态障碍物）
        
        Args:
            filename: 环境文件路径
            
        Returns:
            bool: 读取是否成功
        """
        print(f"\n[2/3] 读取环境信息...")
        print(f"    文件: {os.path.basename(filename)}")
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 初始化数据结构
            left_boundary = []
            right_boundary = []
            centerline = []
            obstacles = []
            vehicle_params = {}
            trailer_params = []
            
            # 解析状态
            section = None
            
            for line in lines:
                line = line.strip()
                
                # 统一转大写进行匹配，提高鲁棒性
                line_upper = line.upper()
                
                # 跳过空行和注释
                if not line or line.startswith('#'):
                    if 'LEFT_BOUNDARY' in line_upper:
                        section = 'left_boundary'
                        # print(f"    [DEBUG] Switch to section: {section}")
                    elif 'RIGHT_BOUNDARY' in line_upper:
                        section = 'right_boundary'
                        # print(f"    [DEBUG] Switch to section: {section}")
                    elif 'CENTER_LINE' in line_upper or ('CENTER' in line_upper and 'LINE' in line_upper):
                        section = 'centerline'
                        # print(f"    [DEBUG] Switch to section: {section}")
                    elif 'OBSTACLES' in line_upper:
                        section = 'obstacles'
                        # print(f"    [DEBUG] Switch to section: {section}")
                    elif 'VEHICLE_PARAMS' in line_upper:
                        section = 'vehicle_params'
                    elif 'TRAILER_PARAMS' in line_upper:
                        section = 'trailer_params'
                    # 兼容通用边界标签，默认存入 left_boundary
                    elif 'BOUNDARY' in line_upper and 'LEFT' not in line_upper and 'RIGHT' not in line_upper:
                        section = 'left_boundary'
                        # print(f"    [DEBUG] Switch to section: {section} (generic BOUNDARY)")
                    continue
                
                # 解析数据
                if section == 'left_boundary':
                    parts = line.split(',')
                    if len(parts) == 2:
                        left_boundary.append([float(parts[0]), float(parts[1])])
                
                elif section == 'right_boundary':
                    parts = line.split(',')
                    if len(parts) == 2:
                        right_boundary.append([float(parts[0]), float(parts[1])])
                
                elif section == 'centerline':
                    parts = line.split(',')
                    if len(parts) == 2:
                        centerline.append([float(parts[0]), float(parts[1])])
                
                elif section == 'obstacles':
                    parts = line.split(',')
                    if len(parts) == 5:
                        obstacles.append({
                            'x': float(parts[0]),
                            'y': float(parts[1]),
                            'yaw': float(parts[2]),
                            'length': float(parts[3]),
                            'width': float(parts[4]),
                            'type': 'static'
                        })
                
                elif section == 'vehicle_params':
                    parts = line.split(',')
                    if len(parts) == 3:
                        vehicle_params = {
                            'tractor_length': float(parts[0]),
                            'tractor_width': float(parts[1]),
                            'num_trailers': int(parts[2])
                        }
                
                elif section == 'trailer_params':
                    parts = line.split(',')
                    if len(parts) == 3:
                        trailer_params.append({
                            'index': int(parts[0]),
                            'length': float(parts[1]),
                            'width': float(parts[2])
                        })
            
            # [Safety] 过滤：如果边界点出现在中心线中，则移除（防止中心线被误当成边界）
            # 使用简单的距离阈值过滤，或者精确匹配
            if centerline and (left_boundary or right_boundary):
                # 构建 KDTree 或简单循环过滤
                # 这里假设点数不多，使用简单距离过滤
                center_arr = np.array(centerline)
                
                def filter_boundary(boundary_list):
                    if not boundary_list:
                        return []
                    filtered = []
                    b_arr = np.array(boundary_list)
                    # 如果边界点距离任意中心线点小于 0.1m，则认为是误报
                    for p in boundary_list:
                        dists = np.sqrt(np.sum((center_arr - np.array(p))**2, axis=1))
                        if np.min(dists) > 0.1:
                            filtered.append(p)
                    return filtered

                if left_boundary:
                    len_before = len(left_boundary)
                    left_boundary = filter_boundary(left_boundary)
                    if len(left_boundary) < len_before:
                        print(f"    [WARN] Filtered {len_before - len(left_boundary)} centerline points from left_boundary")

                if right_boundary:
                    len_before = len(right_boundary)
                    right_boundary = filter_boundary(right_boundary)
                    if len(right_boundary) < len_before:
                        print(f"    [WARN] Filtered {len_before - len(right_boundary)} centerline points from right_boundary")

            self.environment = {
                'left_boundary': np.array(left_boundary),
                'right_boundary': np.array(right_boundary),
                'centerline': np.array(centerline) if centerline else None,
                'obstacles': obstacles,
                'vehicle_params': vehicle_params,
                'trailer_params': trailer_params
            }
            
            # 更新车辆参数
            if vehicle_params:
                self.tractor_length = vehicle_params['tractor_length']
                self.tractor_width = vehicle_params['tractor_width']
            
            print(f"    [OK] 读取成功")
            print(f"      - 左边界点数: {len(left_boundary)}")
            print(f"      - 右边界点数: {len(right_boundary)}")
            print(f"      - 中心线点数: {len(centerline)}")
            print(f"      - 静态障碍物: {len(obstacles)}个")
            
            return True
            
        except Exception as e:
            print(f"    [ERR] 读取失败: {e}")
            return False

    def load_dynamic_obstacles(self, filename):
        """读取动态障碍物未来轨迹点序列。

        支持格式：CSV，表头包含至少：
            step,time,x,y,yaw,velocity,length,width

        当前实现按“一个文件一个动态障碍物”读取；若后续需要多障碍物，可扩展为分段/多文件。
        """
        print(f"\n[3/3] 读取动态障碍物...")
        print(f"    文件: {os.path.basename(filename)}")

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                raw_lines = [ln.strip() for ln in f.readlines()]

            lines = [ln for ln in raw_lines if ln]

            # 跳过注释行，找到表头
            header_line_idx = None
            for i, ln in enumerate(lines):
                if ln.startswith('#'):
                    continue
                if 'time' in ln and 'x' in ln and 'y' in ln:
                    header_line_idx = i
                    break
            if header_line_idx is None:
                raise ValueError('未找到CSV表头')

            header = [h.strip() for h in lines[header_line_idx].split(',')]
            col_idx = {name: header.index(name) for name in header}
            required = ['time', 'x', 'y', 'yaw', 'velocity', 'length', 'width']
            for name in required:
                if name not in col_idx:
                    raise ValueError(f"缺少列: {name}")

            time_list = []
            x_list = []
            y_list = []
            yaw_list = []
            v_list = []
            length_list = []
            width_list = []

            for ln in lines[header_line_idx + 1:]:
                if ln.startswith('#'):
                    continue
                parts = [p.strip() for p in ln.split(',')]
                if len(parts) < len(header):
                    continue
                time_list.append(float(parts[col_idx['time']]))
                x_list.append(float(parts[col_idx['x']]))
                y_list.append(float(parts[col_idx['y']]))
                yaw_list.append(float(parts[col_idx['yaw']]))
                v_list.append(float(parts[col_idx['velocity']]))
                length_list.append(float(parts[col_idx['length']]))
                width_list.append(float(parts[col_idx['width']]))

            if len(time_list) < 2:
                raise ValueError('动态障碍物轨迹点过少')

            obs = {
                'time': np.array(time_list, dtype=float),
                'x': np.array(x_list, dtype=float),
                'y': np.array(y_list, dtype=float),
                'yaw': np.array(yaw_list, dtype=float),
                'velocity': np.array(v_list, dtype=float),
                'length': float(np.median(length_list)),
                'width': float(np.median(width_list)),
                'type': 'dynamic'
            }
            self.dynamic_obstacles = [obs]

            print(f"    [OK] 读取成功")
            print(f"      - 动态障碍物数量: {len(self.dynamic_obstacles)}")
            print(f"      - 轨迹点数: {len(obs['time'])}")
            print(f"      - 时间范围: {obs['time'][0]:.2f}s ~ {obs['time'][-1]:.2f}s")
            return True

        except Exception as e:
            print(f"    [ERR] 读取失败: {e}")
            self.dynamic_obstacles = []
            return False
    
    def _extend_reference_segment_beyond_goal(self, reference_segment, extension_length_m):
        """按参考线末端切向延拓一段参考路径。"""
        if reference_segment is None or len(reference_segment) < 2:
            return reference_segment
        extension_length_m = float(max(0.0, extension_length_m))
        if extension_length_m <= 1e-6:
            return reference_segment

        out = [list(map(float, p)) for p in reference_segment]
        p_prev = np.array(out[-2], dtype=float)
        p_last = np.array(out[-1], dtype=float)
        dxy = p_last - p_prev
        seg_len = float(np.hypot(dxy[0], dxy[1]))
        if seg_len <= 1e-9:
            return out

        tangent = dxy / seg_len
        spacing = float(max(0.5, min(2.0, seg_len)))
        n_extra = int(max(1, math.ceil(extension_length_m / spacing)))
        for i in range(1, n_extra + 1):
            p = p_last + float(i * spacing) * tangent
            out.append([float(p[0]), float(p[1])])
        return out
    
    def plan(self, output_filename='output/local_trajectory.csv', anim_filename=None):
        """
        执行局部路径规划
        模仿Lattice.py中的做法：每次只执行一小段，然后从新位置重新规划
        
        Returns:
            (planning_results, local_execution_path): 规划结果和实际执行路径
        """
        print("\n" + "=" * 70)
        print(" 开始局部路径规划 (滚动规划模式)")
        print("=" * 70)
        
        if self.global_trajectory is None:
            print("未加载全局轨迹")
            return None, []
        
        if self.environment is None:
            print("未加载环境信息")
            return None, []
        
        num_steps = len(self.global_trajectory['step'])
        replan_steps = max(1, int(round(self.config['replan_interval'] / self.config['dt'])))
        lookahead_steps = int(self.config['planning_horizon'] / self.config['dt'])
        
        local_planning_results = []
        local_execution_path = []  # 实际执行的路径

        # “无有效触须”处理：
        # - 优先沿上一次有效触须继续行走
        # - 若连续无有效触须次数超过阈值，则刹停一段时间，再进行重规划；若仍无效则循环
        no_valid_streak = 0
        stop_threshold = int(self.config.get('no_valid_stop_threshold', 3))
        stop_duration_s = float(self.config.get('no_valid_stop_duration_s', 2.0))
        last_valid_plan = None  # {'tractor_traj','trailer_trajs','speed_profile','path_s','cursor'}

        # 当前纵向速度（用于 ST 速度规划的初始条件）
        current_speed = float(self.config.get('initial_speed', 0.0))
        
        # 累计仿真时间（用于 ST 速度规划与动态障碍物对齐）
        elapsed_time = 0.0

        # 当前参考路径索引（用于提取前方参考段）
        current_step = 0
        replan_count = 0
        
        # 初始化当前位置（从全局轨迹起点开始）
        current_pos = np.array([
            float(self.global_trajectory['tractor_x'][0]),
            float(self.global_trajectory['tractor_y'][0])
        ])
        current_yaw = float(self.global_trajectory['tractor_yaw'][0])
        
        # 初始化挂车状态
        tractor_state = {'x': current_pos[0], 'y': current_pos[1], 'yaw': current_yaw}
        current_trailer_states = self.kinematics.initialize_trailers(tractor_state)
        
        # 添加起点
        local_execution_path.append({
            'x': current_pos[0],
            'y': current_pos[1],
            'yaw': current_yaw,
            'v': float(current_speed),
            't': float(elapsed_time),
            'trailers': [
                {'x': float(s['x']), 'y': float(s['y']), 'yaw': float(s['yaw'])}
                for s in (current_trailer_states or [])
            ],
        })
        
        print(f"\n规划参数:")
        print(f"  全局轨迹总步数: {num_steps}")
        effective_replan_interval = replan_steps * self.config['dt']
        effective_replan_hz = 1.0 / effective_replan_interval if effective_replan_interval > 0 else float('inf')
        print(f"  重规划间隔: {replan_steps}步 ({effective_replan_interval:.3f}s, ~{effective_replan_hz:.1f}Hz)")
        print(f"  前瞻距离: {lookahead_steps}步 ({self.config['planning_horizon']}s)")
        print(f"  规划模式: 滚动规划(从实际执行位置重新规划)")
        
        # 迭代上限：原先按 num_steps/replan_steps 会在低速/绕行时过早触发上限。
        # 改为“可配置 + 更宽松”的重规划预算，并加入“长期无进展”保护避免死循环。
        base_replans = int(math.ceil(float(num_steps) / float(max(1, replan_steps))))
        max_replans = self.config.get('max_replans', None)
        if max_replans is None:
            factor = float(self.config.get('max_replans_factor', 5.0))
            min_budget = int(self.config.get('max_replans_min', 500))
            max_replans = int(max(min_budget, base_replans + 50, math.ceil(base_replans * factor)))
        max_iterations = int(max_replans)
        iteration = 0

        best_dist_to_goal = float('inf')
        no_progress_replans = 0
        stagnation_limit = int(self.config.get('stagnation_replans', 2000))
        progress_eps = float(self.config.get('progress_eps', 0.05))
        
        while iteration < max_iterations:
            iteration += 1
            replan_count += 1
            
            # 开始计时这一次重规划
            replan_start_time = time.time()
            
            # ★关键改变★：使用实际当前位置，而不是全局轨迹[current_step]
            # current_pos 和 current_yaw 来自上一次触须执行的终点
            
            # 找到当前位置在全局轨迹上最近的点（用于提取参考路径段）
            distances = np.sqrt(
                (self.global_trajectory['tractor_x'] - current_pos[0])**2 + 
                (self.global_trajectory['tractor_y'] - current_pos[1])**2
            )
            current_step = int(np.argmin(distances))
            
            # 提取前方参考路径段
            end_step = min(current_step + lookahead_steps, num_steps)
            
            remaining_steps = end_step - current_step
            terminal_stop_remaining_steps = int(max(2, self.config.get('terminal_path_stop_remaining_steps', 10)))
            if remaining_steps < terminal_stop_remaining_steps:
                print(f"   剩余路径太短 ({remaining_steps}步)，结束规划")
                break
            
            # 检查是否接近终点
            goal_pos = np.array([
                self.global_trajectory['tractor_x'][-1],
                self.global_trajectory['tractor_y'][-1]
            ])
            dist_to_goal = np.linalg.norm(current_pos - goal_pos)

            # 进展监测：长时间没有“距离终点”改善，则认为卡住，避免无限重规划
            if dist_to_goal < best_dist_to_goal - progress_eps:
                best_dist_to_goal = float(dist_to_goal)
                no_progress_replans = 0
            else:
                no_progress_replans += 1
                if no_progress_replans >= stagnation_limit:
                    print(f"  [ERR] 长时间无进展（{no_progress_replans}次重规划距离终点未显著减小），提前结束。")
                    break

            if dist_to_goal < 2.0:  # 距离终点2米以内
                print(f"  [OK] 已到达终点 (距离={dist_to_goal:.2f}m)")
                break
            
            reference_segment = []
            sample_interval = max(2, int(lookahead_steps / 50))  # 加密参考点，避免弯道抽稀
            for i in range(current_step, end_step, sample_interval):
                reference_segment.append([
                    self.global_trajectory['tractor_x'][i],
                    self.global_trajectory['tractor_y'][i]
                ])
            
            # 确保包含终点
            if len(reference_segment) > 0:
                last_point = [self.global_trajectory['tractor_x'][end_step-1],
                             self.global_trajectory['tractor_y'][end_step-1]]
                if reference_segment[-1] != last_point:
                    reference_segment.append(last_point)

            # 若已经接近全局轨迹终点，但场景仍需要继续跟车，则按末端切向延拓参考线，
            # 避免因为全局轨迹文件过短而让速度规划过早减速。
            if (
                bool(self.config.get('terminal_path_extension_enable', True))
                and end_step >= num_steps
                and len(reference_segment) >= 2
            ):
                reference_segment = self._extend_reference_segment_beyond_goal(
                    reference_segment,
                    float(self.config.get('terminal_path_extension_m', 0.0))
                )
            
            print(f"\n[重规划 {replan_count}]")
            print(f"  当前步数: {current_step}/{num_steps} (t={elapsed_time:.1f}s)")
            print(f"  当前位置: ({current_pos[0]:.2f}, {current_pos[1]:.2f})")
            print(f"  当前航向: {current_yaw:.2f} rad ({math.degrees(current_yaw):.1f} deg)")
            print(f"  参考点数: {len(reference_segment)}")
            
            # 两阶段触须生成：
            # 1) 先生成“中心5条”以减少无意义计算、避免过度贴边
            current_time_index = int(round(elapsed_time / float(self.config['dt'])))

            replan_result = run_single_replan(
                planner=self,
                current_pos=[current_pos[0], current_pos[1]],
                current_yaw=current_yaw,
                current_speed=current_speed,
                current_trailer_states=current_trailer_states,
                elapsed_time=elapsed_time,
                reference_segment=reference_segment,
                include_all_tentacles=True,
            )
            replan_time = time.time() - replan_start_time
            tentacles = list(replan_result.tentacles or [])
            generate_time = float(
                replan_result.timing.get('phase1_gen_eval', replan_result.timing.get('generate_and_eval', 0.0))
            )
            evaluate_time = float(replan_result.timing.get('evaluate', 0.0))
            select_start = time.time()
            
            if replan_result.success:
                best_tentacle = replan_result.best_tentacle
                speed_profile = replan_result.speed_profile
                self.tentacles = tentacles
                valid_count = replan_result.valid_count
                total_count = replan_result.total_count
                phase = replan_result.phase
                if replan_result.current_step is not None:
                    current_step = int(replan_result.current_step)
                self.reference_segment = replan_result.reference_segment
                
                # 更新统计信息
                self.perf_stats['replan_count'] += 1
                self.perf_stats['total_replan_time'] += replan_time
                self.perf_stats['valid_tentacles'] += valid_count
                self.perf_stats['total_tentacles'] += total_count
                if phase in self.perf_stats['phase_counts']:
                    self.perf_stats['phase_counts'][phase] += 1
            else:
                print(f"    [WARN] 单次重规划失败: {replan_result.reason}")
                # 如果重规划失败，也执行保守停车
                best_tentacle = None
                speed_profile = None

            # 与原主流程一致：只要有最优触须和速度剖面即可执行
            if best_tentacle is None or speed_profile is None:
                print(f"    [Fallback] 本轮单次重规划未产出可执行的路径+速度，进入回退逻辑。")
                best_tentacle = None
                speed_profile = None
            
            # ==========================================
            # 策略一致性检查 (Hysteresis)
            # ==========================================
            # 如果上一帧有有效计划，且剩余部分仍然有效，且代价没有比当前最优差太多，则优先保持
            if best_tentacle and last_valid_plan is not None:
                # 上一帧计划是从 prev_pos 开始的。当前 pos 是 prev_pos 走了 replan_steps 后的位置。
                # 所以我们在 last_valid_plan 中的位置应该是 replan_steps
                # FIX: tractor_traj 的点密度是按距离（0.5m）分布的，而 replan_steps 是按时间（dt）分布的
                # 不能直接用 replan_steps 作为索引，必须通过 s (弧长) 来映射
                s_traveled = 0.0
                sp = last_valid_plan.get('speed_profile')
                # 关键：上一条计划实际执行了多少（cursor），不一定等于 replan_steps
                executed_steps_prev = int(max(0, last_valid_plan.get('cursor', 0)))
                if sp and 's' in sp and len(sp['s']) > executed_steps_prev:
                    s_traveled = float(sp['s'][executed_steps_prev])
                
                path_s = last_valid_plan.get('path_s')
                if path_s is not None:
                    # 找到第一个 s >= s_traveled 的索引
                    prev_idx = int(np.searchsorted(path_s, s_traveled))
                else:
                    prev_idx = int(executed_steps_prev)

                traj_len = len(last_valid_plan['tractor_traj'])
                
                if prev_idx < traj_len - 10: # 至少还有点余量
                    # 构造保留的上一帧路径
                    # FIX: 必须将当前位置 prepend 到轨迹开头，消除因离散化导致的“瞬移”
                    curr_pt = [float(current_pos[0]), float(current_pos[1]), float(current_yaw)]
                    retained_tractor_trajectory = [curr_pt] + last_valid_plan['tractor_traj'][prev_idx:]
                    
                    retained_trailer_trajectories = []
                    for j, t_traj in enumerate(last_valid_plan['trailer_trajs']):
                        if current_trailer_states and j < len(current_trailer_states):
                            ts = current_trailer_states[j]
                            t_pt = [float(ts['x']), float(ts['y']), float(ts['yaw'])]
                            retained_trailer_trajectories.append([t_pt] + t_traj[prev_idx:])
                        else:
                            retained_trailer_trajectories.append(t_traj[prev_idx:])
                    
                    retained_tentacle = {
                        'lateral_offset': last_valid_plan.get('lateral_offset', 0.0),
                        'tractor_trajectory': retained_tractor_trajectory,
                        'trailer_trajectories': retained_trailer_trajectories,
                        'cost': 0.0,
                        'is_valid': True,
                        'type': 'retained'
                    }
                    
                    # 重新评估代价
                    retained_cost = evaluate_tentacle_cost(self, retained_tentacle, current_time_index)
                    
                    if retained_tentacle['is_valid']:
                        # 获取分项代价，用于决策逻辑
                        retained_col_cost, _, _ = check_collision(self, retained_tentacle, current_time_index)
                        retained_bnd_cost, _ = check_road_boundary(self, retained_tentacle)
                        hysteresis_switch_margin = float(self.config.get('hysteresis_switch_margin', 50.0))
                        low_speed_switch_margin = float(self.config.get('hysteresis_low_speed_switch_margin', 5.0))
                        min_speed_for_hold = float(self.config.get('hysteresis_min_speed_for_obstacle_hold', 2.0))
                        max_obs_cost_for_hold = float(self.config.get('hysteresis_max_obs_cost_for_hold', 0.08))
                        
                        # 用户逻辑：
                        # 1. 如果碰道路边界 (retained_bnd_cost > 0)，应该换道路 -> 使用原始高代价 (retained_cost)
                        # 2. 如果碰动态障碍物 (leg_col_cost > 0) 但没碰边界，保持原道路 -> 忽略碰撞代价
                        
                        effective_retained_cost = retained_cost
                        can_ignore_obstacle_cost = (
                            retained_bnd_cost <= 1e-3
                            and retained_col_cost > 1e-3
                            and retained_col_cost <= max_obs_cost_for_hold
                            and float(current_speed) >= min_speed_for_hold
                        )
                        if can_ignore_obstacle_cost:
                            # 仅受障碍物影响，忽略碰撞代价以保持路径
                            effective_retained_cost -= (retained_col_cost * 1000.0)
                            # 打印调试信息
                            print(f"  [Hysteresis] 忽略障碍物碰撞代价: Original={retained_cost:.1f} -> Effective={effective_retained_cost:.1f}")
                        elif retained_bnd_cost <= 1e-3 and retained_col_cost > 1e-3:
                            print(
                                f"  [Hysteresis] 释放保留路径: col={retained_col_cost:.3f}, "
                                f"v={float(current_speed):.2f}m/s"
                            )

                        current_best_cost = best_tentacle['cost']
                        switch_margin = hysteresis_switch_margin if float(current_speed) >= min_speed_for_hold else low_speed_switch_margin
                        # 切换代价 (Switch Cost): 新路径必须比旧路径好 50 分以上才切换
                        if effective_retained_cost < current_best_cost + switch_margin:
                            print(f"  [KEEP] 保持上一帧路径 (Retained={effective_retained_cost:.1f} (raw {retained_cost:.1f}) vs New={current_best_cost:.1f}, margin={switch_margin:.1f})")
                            best_tentacle = retained_tentacle
                            # 与原主流程保持一致：保留路径后，按当前状态重新进行速度规划
                            if last_valid_plan is not None:
                                speed_profile, kept_traj = build_speed_and_trajectory(
                                    planner=self,
                                    best_tentacle=best_tentacle,
                                    current_speed=float(current_speed),
                                    elapsed_time=float(elapsed_time),
                                )
                                replan_result.speed_profile = speed_profile
                                replan_result.trajectory = list(kept_traj or [])
                        else:
                            print(f"  [SWITCH] 切换到新路径 (Retained={effective_retained_cost:.1f} (raw {retained_cost:.1f}) vs New={current_best_cost:.1f}, margin={switch_margin:.1f})")

            select_time = time.time() - select_start

            if best_tentacle and best_tentacle.get('swing_active'):
                print(f"  [STRATEGY] 弯道借道策略激活 (Target Offset -> Outside)")

            # 记录“无任何有效触须”的重规划点（不直接中断，便于后续用测试脚本回放定位）
            if valid_count == 0:
                print(
                    f"      [ERR] 无任何有效触须"
                    f" (重规划={replan_count}, step={current_step}/{num_steps}, t={elapsed_time:.3f}s, pos=({current_pos[0]:.2f},{current_pos[1]:.2f}))"
                )
                # 打印每条触须的拒绝原因
                for idx, t in enumerate(tentacles):
                    reason = t.get('reject_reason', 'unknown')
                    lat_off = t.get('lateral_offset', 0.0)
                    print(f"        Tentacle {idx} (lat={lat_off:.2f}): {reason}")

            # 如果没有找到任何触须（生成失败或全部无效且未选择），则执行 fallback
            if not best_tentacle:
                no_valid_streak += 1

                # 只要还有上一条有效路径，就优先沿原路径继续行走，
                # 避免因短时评估抖动而在跟车场景中原地刹停到 0。
                # 真正的原地停等仅用于“没有任何历史可回退”的情况。
                if no_valid_streak > stop_threshold and last_valid_plan is None:
                    dt = float(self.config['dt'])
                    stop_steps = int(round(stop_duration_s / dt))
                    stop_steps = max(1, stop_steps)

                    print(
                        f"  [STOP] 连续无有效触须={no_valid_streak} (> {stop_threshold})，"
                        f"刹停{stop_duration_s:.1f}s 后再重规划"
                    )

                    # 刹停期间：位置不变，速度为 0，时间推进
                    for i in range(1, stop_steps + 1, 2):
                        local_execution_path.append({
                            'x': float(current_pos[0]),
                            'y': float(current_pos[1]),
                            'yaw': float(current_yaw),
                            'v': 0.0,
                            't': float(elapsed_time + float(i) * dt),
                            'trailers': [
                                {'x': float(s['x']), 'y': float(s['y']), 'yaw': float(s['yaw'])}
                                for s in (current_trailer_states or [])
                            ],
                        })
                    elapsed_time += float(stop_steps) * dt
                    current_speed = 0.0
                    self.last_exec_acc = 0.0
                    continue

                # 否则：沿上一次有效路径继续行走，直到重规划重新找到有效触须
                if last_valid_plan is not None:
                    dt = float(self.config['dt'])
                    tractor_traj = last_valid_plan['tractor_traj']
                    trailer_trajs = last_valid_plan['trailer_trajs']
                    speed_profile = last_valid_plan['speed_profile']
                    path_s = last_valid_plan['path_s']
                    cursor = int(last_valid_plan.get('cursor', 0))
                    s_list = speed_profile.get('s', [])
                    v_list = speed_profile.get('v', [])

                    max_idx = min(cursor + int(replan_steps), max(0, len(s_list) - 1))
                    delta_steps = int(max_idx - cursor)
                    if delta_steps <= 0:
                        print("  [WARN] 上一次有效路径已执行到末端，无法继续沿该路径前进")
                    else:
                        for k in range(1, delta_steps + 1, 2):
                            i = cursor + k
                            s_i = float(s_list[i])
                            v_i = float(v_list[i])
                            i0, i1, alpha = self._find_segment_by_s(path_s, s_i)
                            point = self._interp_traj_by_index(tractor_traj, i0, i1, alpha)

                            trailers_point = []
                            for j in range(self.num_trailers):
                                trailer_point = self._interp_traj_by_index(trailer_trajs[j], i0, i1, alpha)
                                trailers_point.append({
                                    'x': float(trailer_point[0]),
                                    'y': float(trailer_point[1]),
                                    'yaw': float(trailer_point[2]),
                                })

                            local_execution_path.append({
                                'x': float(point[0]),
                                'y': float(point[1]),
                                'yaw': float(point[2]),
                                'v': float(v_i),
                                't': float(elapsed_time + float(k) * dt),
                                'trailers': trailers_point,
                            })

                        s_exec = float(s_list[max_idx])
                        v_exec = float(v_list[max_idx])
                        i0, i1, alpha = self._find_segment_by_s(path_s, s_exec)
                        executed_point = self._interp_traj_by_index(tractor_traj, i0, i1, alpha)
                        current_pos = np.array([float(executed_point[0]), float(executed_point[1])])
                        current_yaw = float(executed_point[2])
                        current_speed = float(v_exec)
                        if max_idx >= 1:
                            self.last_exec_acc = float(
                                (float(v_list[max_idx]) - float(v_list[max_idx - 1])) / dt
                            )
                        else:
                            self.last_exec_acc = float(speed_profile.get('a_exec', speed_profile.get('a', 0.0)))
                        elapsed_time += float(delta_steps) * dt

                        current_trailer_states = []
                        for j in range(self.num_trailers):
                            trailer_point = self._interp_traj_by_index(trailer_trajs[j], i0, i1, alpha)
                            current_trailer_states.append({
                                'x': float(trailer_point[0]),
                                'y': float(trailer_point[1]),
                                'yaw': float(trailer_point[2])
                            })

                        last_valid_plan['cursor'] = int(max_idx)
                        print(
                            f"  -> 沿上一次有效路径继续执行{delta_steps}步: "
                            f"pos=({current_pos[0]:.2f},{current_pos[1]:.2f}), v={current_speed:.2f}m/s "
                            f"(连续无有效触须={no_valid_streak}/{stop_threshold})"
                        )
                    continue

                # 没有历史有效路径：退化为沿全局路径小步前进
                print(f"  [WARN] 无历史有效路径可回退，沿全局路径小步前进")
                fallback_step = int(current_step) if current_step is not None else int(np.argmin(distances))
                next_step = min(fallback_step + 10, num_steps - 1)

                # 计算实际距离，避免因步长过大导致动画速度过快
                gx_curr = float(self.global_trajectory['tractor_x'][fallback_step])
                gy_curr = float(self.global_trajectory['tractor_y'][fallback_step])
                gx_next = float(self.global_trajectory['tractor_x'][next_step])
                gy_next = float(self.global_trajectory['tractor_y'][next_step])
                dist_fallback = math.hypot(gx_next - gx_curr, gy_next - gy_curr)

                # 设定回退速度（例如 2m/s 或当前速度），计算所需时间
                # 确保速度不为0，且不超过 v_max
                fallback_speed = max(2.0, float(current_speed))
                fallback_speed = min(fallback_speed, float(self.config['v_max']))
                time_fallback = dist_fallback / fallback_speed if fallback_speed > 0.1 else 1.0

                current_pos = np.array([gx_next, gy_next])
                current_yaw = float(self.global_trajectory['tractor_yaw'][next_step])
                tractor_state = {'x': current_pos[0], 'y': current_pos[1], 'yaw': current_yaw}
                current_trailer_states = self.kinematics.initialize_trailers(tractor_state)

                local_execution_path.append({
                    'x': current_pos[0],
                    'y': current_pos[1],
                    'yaw': current_yaw,
                    'v': float(fallback_speed),
                    't': float(elapsed_time + time_fallback),
                    'trailers': [
                        {'x': float(s['x']), 'y': float(s['y']), 'yaw': float(s['yaw'])}
                        for s in (current_trailer_states or [])
                    ],
                })
                elapsed_time += time_fallback
                self.last_exec_acc = 0.0
                continue

            # 有有效触须：正常执行（并重置连续计数）
            no_valid_streak = 0

            if best_tentacle:
                print(f"  [OK] 有效触须: {valid_count}/{len(tentacles)}")
                print(f"  [OK] 最优触须: 横向偏移={best_tentacle['lateral_offset']:.2f}m, "
                      f"代价={best_tentacle['cost']:.2f}, "
                      f"有效={best_tentacle['is_valid']}")
                
                # 计算总耗时
                total_time = generate_time + evaluate_time + select_time
                print(f"   耗时统计:")
                print(f"      - 生成触须: {generate_time*1000:.2f}ms ({generate_time/total_time*100:.1f}%)")
                print(f"      - 评估代价: {evaluate_time*1000:.2f}ms ({evaluate_time/total_time*100:.1f}%)")
                print(f"      - 选择触须: {select_time*1000:.2f}ms ({select_time/total_time*100:.1f}%)")
                print(f"      - 总耗时: {total_time*1000:.2f}ms")
                
                local_planning_results.append({
                    'step': current_step,
                    'time': elapsed_time,
                    'tentacles': tentacles,
                    'best_tentacle': best_tentacle,
                    'current_pos': current_pos,
                    'current_yaw': current_yaw,
                    'reference_segment': reference_segment
                })
                
                # ★关键★：执行最优触须的一小段，然后从终点重新规划
                # 模仿Lattice.py: traj_point = traj_points_opt[1]
                best_tractor_trajectory = best_tentacle['tractor_trajectory']
                best_trailer_trajectories = best_tentacle['trailer_trajectories']

                # 单次重规划接口已在内部完成路径+速度规划，主循环只消费其输出
                local_planning_results[-1]['speed_profile'] = {
                    'a': float(speed_profile.get('a', 0.0)),
                    't': speed_profile.get('t', []),
                    's': speed_profile.get('s', []),
                    'v': speed_profile.get('v', []),
                }
                best_tentacle['speed_profile'] = local_planning_results[-1]['speed_profile']

                # 牵引车路径弧长，用于把 s(t) 映射回轨迹点
                best_traj_np = np.array(best_tractor_trajectory, dtype=float)
                path_xy = best_traj_np[:, :2]
                path_s = self._compute_arc_length_xy(path_xy)
                
                # 计算实际执行的步数（用“走完路径的一定比例”来决定何时重规划）
                # 默认执行 1/2 路程再重规划，避免高频重规划导致每次只走极短距离。
                s_list = speed_profile.get('s', []) or []
                max_speed_step = max(0, len(s_list) - 1)

                exec_ratio = float(self.config.get('execute_progress_ratio', 0.5))
                if exec_ratio <= 0.0 or max_speed_step <= 0:
                    # 退化回旧策略：按 replan_interval 执行
                    execution_step = min(replan_steps, max_speed_step)
                    execution_step = max(1, int(execution_step))
                else:
                    exec_ratio = float(max(1e-6, min(1.0, exec_ratio)))
                    s_total = float(s_list[max_speed_step])
                    s_target = float(exec_ratio * s_total)
                    # 找到第一个 s >= s_target 的时间步
                    idx = int(np.searchsorted(np.asarray(s_list, dtype=float), s_target, side='left'))
                    execution_step = int(max(1, min(idx, max_speed_step)))

                # 记录本次重规划实际执行步数（用于动画避免“回退”观感）
                if local_planning_results:
                    local_planning_results[-1]['execution_step'] = execution_step
                
                # 添加执行的点到路径（降采样）
                for i in range(1, execution_step + 1, 2):  # 每2步采样1个
                    s_i = float(speed_profile['s'][i])
                    v_i = float(speed_profile['v'][i])
                    i0, i1, alpha = self._find_segment_by_s(path_s, s_i)
                    point = self._interp_traj_by_index(best_tractor_trajectory, i0, i1, alpha)

                    trailers_point = []
                    for j in range(self.num_trailers):
                        trailer_point = self._interp_traj_by_index(best_trailer_trajectories[j], i0, i1, alpha)
                        trailers_point.append({
                            'x': float(trailer_point[0]),
                            'y': float(trailer_point[1]),
                            'yaw': float(trailer_point[2]),
                        })

                    local_execution_path.append({
                        'x': float(point[0]),
                        'y': float(point[1]),
                        'yaw': float(point[2]),
                        'v': float(v_i),
                        't': float(elapsed_time + float(i) * float(self.config['dt'])),
                        'trailers': trailers_point,
                    })

                # 更新当前位置为执行后的终点
                s_exec = float(speed_profile['s'][execution_step])
                v_exec = float(speed_profile['v'][execution_step])
                i0, i1, alpha = self._find_segment_by_s(path_s, s_exec)
                executed_point = self._interp_traj_by_index(best_tractor_trajectory, i0, i1, alpha)
                current_pos = np.array([float(executed_point[0]), float(executed_point[1])])
                current_yaw = float(executed_point[2])
                current_speed = float(v_exec)
                if execution_step >= 1:
                    self.last_exec_acc = float(
                        (float(speed_profile['v'][execution_step]) - float(speed_profile['v'][execution_step - 1]))
                        / float(self.config['dt'])
                    )
                else:
                    self.last_exec_acc = float(speed_profile.get('a_exec', speed_profile.get('a', 0.0)))

                # 累计时间推进（用于下一次重规划的动态障碍物对齐）
                elapsed_time += float(execution_step) * float(self.config['dt'])

                # 更新挂车状态（根据执行位置从挂车触须插值得到）
                current_trailer_states = []
                for j in range(self.num_trailers):
                    trailer_point = self._interp_traj_by_index(best_trailer_trajectories[j], i0, i1, alpha)
                    current_trailer_states.append({
                        'x': float(trailer_point[0]),
                        'y': float(trailer_point[1]),
                        'yaw': float(trailer_point[2])
                    })

                print(f"  -> 执行{execution_step}步后新位置: ({current_pos[0]:.2f}, {current_pos[1]:.2f}), v={current_speed:.2f}m/s")
                print(f"    挂车位置已更新 ({len(current_trailer_states)}节)")

                # 记录“上一次有效路径”，供后续无有效触须时回退执行
                last_valid_plan = {
                    'tractor_traj': best_tractor_trajectory,
                    'trailer_trajs': best_trailer_trajectories,
                    'speed_profile': speed_profile,
                    'path_s': path_s,
                    'cursor': int(execution_step),
                    'lateral_offset': best_tentacle.get('lateral_offset', 0.0),
                }
                
            else:
                print(f"  [ERR] 未找到有效触须，沿全局路径小步前进")
                # 如果没有有效触须，沿全局路径前进一小段
                next_step = min(current_step + 10, num_steps - 1)
                current_pos = np.array([
                    float(self.global_trajectory['tractor_x'][next_step]),
                    float(self.global_trajectory['tractor_y'][next_step])
                ])
                current_yaw = float(self.global_trajectory['tractor_yaw'][next_step])
                
                # 重新初始化挂车状态
                tractor_state = {'x': current_pos[0], 'y': current_pos[1], 'yaw': current_yaw}
                current_trailer_states = self.kinematics.initialize_trailers(tractor_state)
                
                local_execution_path.append({
                    'x': current_pos[0],
                    'y': current_pos[1],
                    'yaw': current_yaw,
                    'v': float(current_speed),
                    't': float(elapsed_time),
                    'trailers': [
                        {'x': float(s['x']), 'y': float(s['y']), 'yaw': float(s['yaw'])}
                        for s in (current_trailer_states or [])
                    ],
                })

                elapsed_time += float(next_step - current_step) * float(self.config['dt'])
        
        print("\n" + "=" * 70)
        print(f"[OK] 局部规划完成，共 {replan_count} 次重规划")
        print(f"[OK] 实际执行路径点数: {len(local_execution_path)}")
        print("=" * 70)
        
        return local_planning_results, local_execution_path
    
    def visualize_planning_result(self, planning_results, local_execution_path=None, save_path=None, show_all_tentacles=False):
        """
        可视化局部规划结果（静态图）
        
        Args:
            planning_results: 规划结果列表
            local_execution_path: 局部执行路径
            save_path: 保存路径（可选）
            show_all_tentacles: 是否显示所有触须（默认只显示最优触须）
        """
        print(f"\n🎨 生成静态可视化...")
        
        fig, ax = plt.subplots(figsize=(18, 10))
        
        # 1. 绘制道路边界
        left_boundary = self.environment.get('left_boundary')
        right_boundary = self.environment.get('right_boundary')
        
        if left_boundary is not None and len(left_boundary) > 0:
            # 改为绘制散点，适应非结构化边界
            ax.plot(left_boundary[:, 0], left_boundary[:, 1], 
                   'k.', markersize=2, label='Boundary points', alpha=0.6)
        
        if right_boundary is not None and len(right_boundary) > 0:
            ax.plot(right_boundary[:, 0], right_boundary[:, 1], 
                   'k.', markersize=2, alpha=0.6)
        
        # 1b. 绘制中心线（单向双车道分界线）
        centerline = self.environment.get('centerline')
        if centerline is not None and len(centerline) > 0:
            ax.plot(centerline[:, 0], centerline[:, 1], 
                   'k.', markersize=2, label='Center line points', alpha=0.6, zorder=6)
        
        # 2. 绘制静态障碍物
        for idx, obs in enumerate(self.environment['obstacles']):
            rect = FancyBboxPatch(
                (-obs['length']/2, -obs['width']/2),
                obs['length'],
                obs['width'],
                boxstyle="round,pad=0.05",
                facecolor='red',
                fill=True,
                alpha=0.8,
                edgecolor='none',
                linewidth=0,
                label='Static obstacle' if idx == 0 else ''
            )
            
            t = Affine2D().rotate_around(0, 0, obs['yaw']).translate(obs['x'], obs['y']) + ax.transData
            rect.set_transform(t)
            ax.add_patch(rect)
        
        # 3. 绘制全局路径（虚线）
        ax.plot(self.global_trajectory['tractor_x'], 
               self.global_trajectory['tractor_y'], 
               'b--', linewidth=1.5, label='Global path (tractor)', alpha=0.5)
        
        # 5. 绘制局部执行路径（细实线）
        if local_execution_path is not None and len(local_execution_path) > 0:
            path_x = [p['x'] for p in local_execution_path]
            path_y = [p['y'] for p in local_execution_path]
            ax.plot(path_x, path_y, 
                   'g-', linewidth=0.45, label='Tractor trajectory', 
                   zorder=8)
        
        # 6. 绘制局部规划结果（最优触须及其挂车轨迹 - 仅显示偶数节）
        colors = plt.cm.rainbow(np.linspace(0, 1, len(planning_results)))
        trailer_colors = ['green', 'purple', 'brown', 'cyan', 'red', 'orange', 'blue', 'pink']
        even_trailer_indices = [i for i in range(self.num_trailers) if (i + 1) % 2 == 0]  # 偶数节：1,3,5,7...
        
        for idx, result in enumerate(planning_results):
            best_tentacle = result['best_tentacle']
            if best_tentacle is None:
                continue
            
            # # 绘制最优触须牵引车轨迹（粗实线，突出显示）
            # tractor_traj = np.array(best_tentacle['tractor_trajectory'])
            # ax.plot(tractor_traj[:, 0], tractor_traj[:, 1], 
            #        color=colors[idx], linewidth=2, alpha=0.8,
            #        label=f'Best Tentacle {idx+1}', zorder=9)
            
            # 绘制最优触须的偶数节挂车轨迹
            if best_tentacle.get('trailer_trajectories') and len(best_tentacle['trailer_trajectories']) > 0:
                for j in even_trailer_indices:
                    if j < len(best_tentacle['trailer_trajectories']):
                        trailer_traj = best_tentacle['trailer_trajectories'][j]
                        trailer_array = np.array(trailer_traj)
                        trailer_color = trailer_colors[j % len(trailer_colors)]
                        ax.plot(trailer_array[:, 0], trailer_array[:, 1], 
                               color=trailer_color, linewidth=1.2, alpha=0.6,
                               linestyle='-', label=f'Trailer {j+1} (Plan {idx+1})' if idx < 2 else '', zorder=8)
            
            # 绘制其他有效触须（细线）
            for tentacle in result['tentacles']:
                if tentacle is best_tentacle or not tentacle['is_valid']:
                    continue
                traj = np.array(tentacle['tractor_trajectory'])
                ax.plot(traj[:, 0], traj[:, 1], 
                       color=colors[idx], linewidth=0.8, alpha=0.5,
                       linestyle=':', zorder=5)
            
            # 标记重规划点
            # 改为“空心 + 更透明”的绿色圆点，避免遮挡挂车轨迹
            ax.plot(
                result['current_pos'][0],
                result['current_pos'][1],
                marker='o',
                linestyle='None',
                markersize=8,
                markerfacecolor='none',
                markeredgecolor='lime',
                markeredgewidth=1.5,
                alpha=0.45,
                zorder=10,
            )
        
        # 7. 绘制起点和终点的车辆
        # 起点车辆（浅色）
        start_idx = 0
        tractor_start = self._get_vehicle_state_from_trajectory(
            self.global_trajectory, start_idx)
        self._draw_vehicle_convoy(ax, tractor_start, 
                                 label_prefix='Start ', show_label=True)
        
        # 终点车辆（深色）
        end_idx = len(self.global_trajectory['tractor_x']) - 1
        tractor_end = self._get_vehicle_state_from_trajectory(
            self.global_trajectory, end_idx)
        self._draw_vehicle_convoy(ax, tractor_end, 
                                 label_prefix='End ', show_label=True)
        
        # 标记起点和终点
        start_x = self.global_trajectory['tractor_x'][0]
        start_y = self.global_trajectory['tractor_y'][0]
        end_x = self.global_trajectory['tractor_x'][-1]
        end_y = self.global_trajectory['tractor_y'][-1]
        
        ax.plot(start_x, start_y, 'go', markersize=12, 
               label='', zorder=10, markeredgecolor='white', markeredgewidth=2)
        ax.plot(end_x, end_y, 'r*', markersize=18, 
               label='', zorder=10, markeredgecolor='white', markeredgewidth=2)
        
        # 设置图表
        ax.set_xlabel('X (m)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=14, fontweight='bold')
        ax.set_title('Tentacle-based Local Path Planning', 
                    fontsize=16, fontweight='bold')
        ax.legend(loc='center left', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axis('equal')
        
        # 添加统计信息
        stats_text = f"Planning Statistics:\n"
        stats_text += f"- Replanning count: {len(planning_results)}\n"
        stats_text += f"- Tentacles per plan: {self.config['num_tentacles']}\n"
        stats_text += f"- Planning horizon: {self.config['planning_horizon']}s"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   静态图已保存到: {save_path}")
        
        plt.close()
        print("  [OK] 静态可视化完成")
    
    def create_animation(self, planning_results, local_execution_path=None, save_path=None):
        """
        创建动画展示局部规划过程（GIF格式）
        基于 local_execution_path 的真实时间轴进行渲染，确保播放速度准确。
        """
        if not local_execution_path or not planning_results:
            print("[WARN] 无路径数据，无法生成动画")
            return

        animation_fps = int(self.config.get('animation_fps', 30))
        dt = float(self.config.get('dt', 0.01))
        
        # 1. 计算总时长和总帧数
        # 假设 local_execution_path 是连续的，间隔为 dt
        # 如果 local_execution_path 里的点有 't' 字段，则使用 't'
        has_time = isinstance(local_execution_path[0], dict) and ('t' in local_execution_path[0])
        
        if has_time:
            start_time = float(local_execution_path[0]['t'])
            end_time = float(local_execution_path[-1]['t'])
            total_duration = end_time - start_time
        else:
            start_time = 0.0
            total_duration = len(local_execution_path) * dt
            end_time = total_duration
            
        total_frames = int(max(1, round(total_duration * animation_fps)))
        
        print(f"\n[GIF] 生成动画（GIF）...")
        print(f"  基于真实执行路径生成")
        print(f"  路径点数: {len(local_execution_path)}")
        print(f"  物理时长: {total_duration:.2f}s")
        
        # Calculate average speed
        dist = 0.0
        if len(local_execution_path) > 1:
            p0 = local_execution_path[0]
            pn = local_execution_path[-1]
            dist = math.hypot(pn['x'] - p0['x'], pn['y'] - p0['y'])
        avg_speed = dist / total_duration if total_duration > 0.001 else 0.0
        print(f"  平均速度: {avg_speed:.2f} m/s")

        print(f"  动画帧率: {animation_fps} fps")
        print(f"  总帧数: {total_frames}")

        # 2. 预处理重规划时间索引，以便快速查找当前时刻对应的触须
        # planning_results[i]['time'] 是该次规划的起始物理时间
        replan_times = [r.get('time', 0.0) for r in planning_results]
        if not has_time:
            base_time = replan_times[0] if replan_times else 0.0
            replan_times = [t - base_time for t in replan_times]

        # 3. 准备绘图数据
        # 把 local_execution_path 转成 numpy，后续每帧仅画“已执行轨迹”（牵引车 + 挂车）
        exec_xy = np.array([[p['x'], p['y']] for p in local_execution_path], dtype=float)

        exec_trailer_xy = []
        if local_execution_path and isinstance(local_execution_path[0], dict) and ('trailers' in local_execution_path[0]):
            for j in range(int(self.num_trailers)):
                coords = []
                for p in local_execution_path:
                    trailers = p.get('trailers', []) or []
                    if j < len(trailers):
                        coords.append([float(trailers[j]['x']), float(trailers[j]['y'])])
                    else:
                        coords.append([float('nan'), float('nan')])
                exec_trailer_xy.append(np.array(coords, dtype=float))
        
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # 计算坐标范围
        all_x = []
        all_y = []

        # 全局路径（牵引车）
        if self.global_trajectory is not None:
            if 'tractor_x' in self.global_trajectory and 'tractor_y' in self.global_trajectory:
                all_x.extend(list(self.global_trajectory['tractor_x']))
                all_y.extend(list(self.global_trajectory['tractor_y']))

        # 道路边界 + 中心线
        left_boundary = self.environment.get('left_boundary')
        right_boundary = self.environment.get('right_boundary')
        if left_boundary is not None and len(left_boundary) > 0:
            all_x.extend(list(left_boundary[:, 0]))
            all_y.extend(list(left_boundary[:, 1]))
        if right_boundary is not None and len(right_boundary) > 0:
            all_x.extend(list(right_boundary[:, 0]))
            all_y.extend(list(right_boundary[:, 1]))

        centerline = self.environment.get('centerline')
        if centerline is not None and len(centerline) > 0:
            all_x.extend(list(centerline[:, 0]))
            all_y.extend(list(centerline[:, 1]))

        # 静态障碍物（用外接圆近似加入范围）
        for obs in self.environment.get('obstacles', []):
            try:
                radius = 0.5 * float(np.hypot(obs.get('length', 0.0), obs.get('width', 0.0)))
                cx = float(obs.get('x', 0.0))
                cy = float(obs.get('y', 0.0))
                all_x.extend([cx - radius, cx + radius])
                all_y.extend([cy - radius, cy + radius])
            except Exception:
                continue

        # 动态障碍物（取整个时间序列的外包络，确保画布能覆盖）
        for obs in getattr(self, 'dynamic_obstacles', []) or []:
            try:
                length = float(obs.get('length', 0.0))
                width = float(obs.get('width', 0.0))
                radius = 0.5 * float(np.hypot(length, width))
                xs = np.asarray(obs.get('x', []), dtype=float)
                ys = np.asarray(obs.get('y', []), dtype=float)
                if xs.size > 0 and ys.size > 0:
                    all_x.extend([float(xs.min()) - radius, float(xs.max()) + radius])
                    all_y.extend([float(ys.min()) - radius, float(ys.max()) + radius])
            except Exception:
                continue

        if not all_x or not all_y:
            all_x = [0.0, 1.0]
            all_y = [0.0, 1.0]

        min_x, max_x = float(min(all_x)), float(max(all_x))
        min_y, max_y = float(min(all_y)), float(max(all_y))
        x_range = max(1e-6, max_x - min_x)
        y_range = max(1e-6, max_y - min_y)

        # “扩大一点”：在范围比例 padding 基础上加最小 padding
        x_margin = max(5.0, x_range * 0.1)
        y_margin = max(5.0, y_range * 0.1)
        
        # 预先提取所有时间戳，用于快速查找
        path_times = []
        if has_time:
            path_times = np.array([p['t'] for p in local_execution_path], dtype=float)
            # 归一化到从0开始
            path_times = path_times - start_time
        
        def update(frame_idx):
            """更新每一帧"""
            # 仅在第一帧时初始化背景（道路、障碍物等）
            if frame_idx == 0:
                ax.clear()
                
                # 1. 绘制道路边界
                left_boundary = self.environment.get('left_boundary')
                right_boundary = self.environment.get('right_boundary')
                
                if left_boundary is not None and len(left_boundary) > 0:
                    ax.plot(left_boundary[:, 0], left_boundary[:, 1], 
                           'k.', markersize=2, alpha=0.7, zorder=1, gid='background')
                
                if right_boundary is not None and len(right_boundary) > 0:
                    ax.plot(right_boundary[:, 0], right_boundary[:, 1], 
                           'k.', markersize=2, alpha=0.7, zorder=1, gid='background')
                
                # 1b. 绘制中心线（单向双车道分界线）
                centerline = self.environment.get('centerline')
                if centerline is not None and len(centerline) > 0:
                    centerline_line = ax.plot(centerline[:, 0], centerline[:, 1], 
                           'k.', markersize=2, alpha=0.6, zorder=1, gid='background')[0]
                    centerline_line.set_gid('background')
                
                # 2. 绘制静态障碍物
                for idx, obs in enumerate(self.environment['obstacles']):
                    from matplotlib.patches import FancyBboxPatch
                    from matplotlib.transforms import Affine2D
                    rect = FancyBboxPatch(
                        (-obs['length']/2, -obs['width']/2),
                        obs['length'],
                        obs['width'],
                        boxstyle="round,pad=0.05",
                        facecolor='red',
                        fill=True,
                        alpha=0.7,
                        edgecolor='none',
                        linewidth=0,
                        zorder=2,
                        gid='background'
                    )
                    t = Affine2D().rotate_around(0, 0, obs['yaw']).translate(obs['x'], obs['y']) + ax.transData
                    rect.set_transform(t)
                    ax.add_patch(rect)
                
                # 3. 绘制全局路径（虚线）
                global_path_line = ax.plot(self.global_trajectory['tractor_x'], 
                       self.global_trajectory['tractor_y'], 
                       'b--', linewidth=1.5, alpha=0.4, label='Global path (tractor)', zorder=2)[0]
                global_path_line.set_gid('background')
                
                # 设置坐标范围
                ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
                ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlabel('X (m)', fontsize=13, fontweight='bold')
                ax.set_ylabel('Y (m)', fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
            
            # 当前物理时间（相对于起点）
            t_current = frame_idx / animation_fps
            
            # 1. 计算车辆位置（在 local_execution_path 上插值）
            if has_time and len(path_times) > 1:
                # 使用真实时间戳查找索引
                # searchsorted 返回第一个 >= t_current 的索引
                idx_next = np.searchsorted(path_times, t_current)
                
                if idx_next == 0:
                    idx_curr = 0
                    idx_next = 0
                    alpha = 0.0
                elif idx_next >= len(path_times):
                    idx_curr = len(path_times) - 1
                    idx_next = len(path_times) - 1
                    alpha = 0.0
                else:
                    idx_curr = idx_next - 1
                    t0 = path_times[idx_curr]
                    t1 = path_times[idx_next]
                    if t1 > t0 + 1e-6:
                        alpha = (t_current - t0) / (t1 - t0)
                    else:
                        alpha = 0.0
            else:
                # 降级方案：假设均匀分布
                idx_float = t_current / dt
                idx_int = int(idx_float)
                alpha = idx_float - idx_int
                idx_curr = min(idx_int, len(local_execution_path) - 1)
                idx_next = min(idx_curr + 1, len(local_execution_path) - 1)
            
            p_curr = local_execution_path[idx_curr]
            p_next = local_execution_path[idx_next]
            
            def _interp_angle(a0, a1, t):
                da = math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))
                return a0 + t * da

            # 车辆状态插值
            tractor_x = (1.0 - alpha) * p_curr['x'] + alpha * p_next['x']
            tractor_y = (1.0 - alpha) * p_curr['y'] + alpha * p_next['y']
            tractor_yaw = _interp_angle(p_curr['yaw'], p_next['yaw'], alpha)
            
            tractor_state = {'x': tractor_x, 'y': tractor_y, 'yaw': tractor_yaw}

            # 2. 查找当前时刻对应的重规划结果（用于画触须）
            replan_idx = 0
            for i, rt in enumerate(replan_times):
                if rt <= t_current + 1e-3:
                    replan_idx = i
                else:
                    break
            
            current_plan = planning_results[replan_idx]
            
            # 清除动态元素
            patches_to_remove = [p for p in ax.patches if p.get_gid() != 'background']
            for p in patches_to_remove: p.remove()
            lines_to_remove = [l for l in ax.lines if l.get_gid() != 'background']
            for l in lines_to_remove: l.remove()
            
            # 绘制触须
            best_tentacle = current_plan.get('best_tentacle', None)
            for tentacle in current_plan['tentacles']:
                traj = np.array(tentacle['tractor_trajectory'])
                if tentacle is best_tentacle:
                    continue
                
                if tentacle['is_valid']:
                    l = ax.plot(traj[:, 0], traj[:, 1], 'gray', linewidth=1.5, alpha=0.2, zorder=4)[0]
                else:
                    l = ax.plot(traj[:, 0], traj[:, 1], 'gray', linestyle=':', linewidth=0.8, alpha=0.1, zorder=3)[0]
                l.set_gid('tentacle')

            # 绘制最优触须
            if best_tentacle:
                best_traj = np.array(best_tentacle['tractor_trajectory'])
                l = ax.plot(best_traj[:, 0], best_traj[:, 1], 'lime', linewidth=2, alpha=0.3, linestyle='--', zorder=7)[0]
                l.set_gid('tentacle')
                
                if best_tentacle.get('trailer_trajectories'):
                    even_trailer_indices = [i for i in range(self.num_trailers) if (i + 1) % 2 == 0]
                    trailer_colors = ['green', 'purple', 'brown', 'cyan', 'red', 'orange', 'blue', 'pink']
                    for j in even_trailer_indices:
                        if j < len(best_tentacle['trailer_trajectories']):
                            tt = np.array(best_tentacle['trailer_trajectories'][j])
                            tc = trailer_colors[j % len(trailer_colors)]
                            l = ax.plot(tt[:, 0], tt[:, 1], color=tc, linewidth=1.5, alpha=0.2, linestyle='--', zorder=6)[0]
                            l.set_gid('tentacle')

            # 绘制已执行轨迹
            if idx_curr > 0:
                l = ax.plot(exec_xy[:idx_curr+1, 0], exec_xy[:idx_curr+1, 1], 'lime', linewidth=1.2, alpha=0.8, zorder=8)[0]
                l.set_gid('past_traj')

                # 绘制挂车已执行轨迹（只画偶数节：2/4/6/8...）
                if exec_trailer_xy:
                    even_trailer_indices = [i for i in range(int(self.num_trailers)) if (i + 1) % 2 == 0]
                    trailer_colors = ['green', 'purple', 'brown', 'cyan', 'red', 'orange', 'blue', 'pink']
                    for j in even_trailer_indices:
                        xy = exec_trailer_xy[j][:idx_curr+1]
                        c = trailer_colors[j % len(trailer_colors)]
                        l2 = ax.plot(xy[:, 0], xy[:, 1], color=c, linewidth=1.0, alpha=0.55, zorder=7)[0]
                        l2.set_gid('past_traj')

            # 绘制车辆
            self._draw_vehicle_convoy(ax, tractor_state, show_label=False)
            
            # 绘制挂车
            trailers_curr = p_curr.get('trailers', [])
            trailers_next = p_next.get('trailers', [])
            
            if trailers_curr and len(trailers_curr) == self.num_trailers:
                even_trailer_indices = [i for i in range(self.num_trailers) if (i + 1) % 2 == 0]
                trailer_colors = ['green', 'purple', 'brown', 'cyan', 'red', 'orange', 'blue', 'pink']
                
                for j in even_trailer_indices:
                    t_c = trailers_curr[j]
                    if j < len(trailers_next):
                        t_n = trailers_next[j]
                        tx = (1.0 - alpha) * t_c['x'] + alpha * t_n['x']
                        ty = (1.0 - alpha) * t_c['y'] + alpha * t_n['y']
                        tyaw = _interp_angle(t_c['yaw'], t_n['yaw'], alpha)
                    else:
                        tx, ty, tyaw = t_c['x'], t_c['y'], t_c['yaw']
                    
                    ts = {'x': tx, 'y': ty, 'yaw': tyaw}
                    tc = trailer_colors[j % len(trailer_colors)]
                    self._draw_single_vehicle(ax, ts, self.trailer_length[j], self.trailer_width[j], tc, '')

            # 绘制动态障碍物
            dyn_obs_list = getattr(self, 'dynamic_obstacles', []) or []
            if dyn_obs_list:
                from matplotlib.patches import FancyBboxPatch
                from matplotlib.transforms import Affine2D
                abs_time = start_time + t_current
                for obs in dyn_obs_list:
                    try:
                        ox, oy, oyaw, _ov = self._dynamic_obstacle_state_at_time(obs, abs_time)
                        length = float(obs.get('length', 0.0))
                        width = float(obs.get('width', 0.0))
                        if length <= 1e-6 or width <= 1e-6: continue
                        
                        rect = FancyBboxPatch(
                            (-length / 2.0, -width / 2.0), length, width,
                            boxstyle="round,pad=0.05", facecolor='orange', fill=True,
                            alpha=0.55, edgecolor='none', linewidth=0, zorder=5
                        )
                        t = Affine2D().rotate_around(0, 0, oyaw).translate(ox, oy) + ax.transData
                        rect.set_transform(t)
                        ax.add_patch(rect)
                    except Exception:
                        continue

            ax.set_title(f'Tentacle Local Planning - t={t_current:.2f}s / {total_duration:.2f}s', 
                        fontsize=14, fontweight='bold')
            
            return []
        
        # 创建动画
        anim = FuncAnimation(
            fig,
            update,
            init_func=None,
            frames=total_frames,
            interval=int(round(1000 / max(1, animation_fps))),
            blit=False,
            repeat=True
        )
        
        # 保存为GIF
        if save_path:
            writer = PillowWriter(fps=max(1, animation_fps))
            anim.save(save_path, writer=writer)
            print(f"   动画已保存到: {save_path}")
            print(f"   文件大小: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
        
        plt.close()
        print("  [OK] 动画生成完成")
    
    def save_local_path_to_csv(self, local_execution_path, filename):
        """保存局部执行路径到 CSV 文件（无注释头）。

        输出列：x,y,yaw,(v?),(t?)
        """
        if local_execution_path is None or len(local_execution_path) == 0:
            print("[ERR] 局部执行路径为空，无法保存")
            return False

        try:
            has_v = isinstance(local_execution_path[0], dict) and ('v' in local_execution_path[0])
            has_t = isinstance(local_execution_path[0], dict) and ('t' in local_execution_path[0])

            header = ["x", "y", "yaw"]
            if has_v:
                header.append("v")
            if has_t:
                header.append("t")

            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            with open(filename, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for point in local_execution_path:
                    row = [
                        f"{float(point['x']):.6f}",
                        f"{float(point['y']):.6f}",
                        f"{float(point['yaw']):.6f}",
                    ]
                    if has_v:
                        row.append(f"{float(point.get('v', 0.0)):.6f}")
                    if has_t:
                        row.append(f"{float(point.get('t', 0.0)):.6f}")
                    writer.writerow(row)

            print(f" 局部执行路径 CSV 已保存到: {filename}")
            print(f"   路径点数: {len(local_execution_path)}")
            return True
        except Exception as e:
            print(f" 保存局部执行路径 CSV 失败: {e}")
            return False
    
    def save_complete_trajectory_with_trailers_to_csv(self, planning_results, filename):
        """保存完整轨迹到 CSV 文件（扁平化、无注释头）。

        每行对应一次重规划里的一个 step。
        输出列：replan_idx,step,tractor_x,tractor_y,tractor_yaw,(trailer_0_x,...)
        """
        if planning_results is None or len(planning_results) == 0:
            print(" 规划结果为空，无法保存")
            return False

        try:
            header = ["replan_idx", "step", "tractor_x", "tractor_y", "tractor_yaw"]
            for i in range(self.num_trailers):
                header += [f"trailer_{i}_x", f"trailer_{i}_y", f"trailer_{i}_yaw"]

            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            with open(filename, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)

                for plan_idx, result in enumerate(planning_results):
                    best_tentacle = result.get('best_tentacle', None)
                    if best_tentacle is None:
                        continue

                    tractor_traj = best_tentacle.get('tractor_trajectory', None)
                    trailer_trajs = best_tentacle.get('trailer_trajectories', [])
                    if tractor_traj is None:
                        continue

                    for step in range(len(tractor_traj)):
                        tractor_point = tractor_traj[step]
                        row = [
                            int(plan_idx),
                            int(step),
                            f"{float(tractor_point[0]):.6f}",
                            f"{float(tractor_point[1]):.6f}",
                            f"{float(tractor_point[2]):.6f}",
                        ]

                        for j in range(self.num_trailers):
                            if trailer_trajs is not None and i < len(trailer_trajs) and trailer_trajs[i] is not None and len(trailer_trajs[i]) > 0:
                                trailer_point = trailer_trajs[i][step]
                                row += [
                                    f"{float(trailer_point[0]):.6f}",
                                    f"{float(trailer_point[1]):.6f}",
                                    f"{float(trailer_point[2]):.6f}",
                                ]
                            else:
                                row += ["0.0", "0.0", "0.0"]

                        writer.writerow(row)

            total_points = sum(
                len(r['best_tentacle']['tractor_trajectory'])
                for r in planning_results
                if r.get('best_tentacle', None) is not None
            )

            print(f" 完整轨迹 CSV 已保存到: {filename}")
            print(f"   重规划次数: {len(planning_results)}")
            print(f"   总轨迹点数: {total_points}")
            print(f"   包含: 头车 + {self.num_trailers}节挂车")
            return True
        except Exception as e:
            print(f"[ERR] 保存完整轨迹 CSV 失败: {e}")
            return False

    def save_all_replans_speed_and_trajectory_to_csv(self, planning_results, filename):
        """把所有重规划的“速度 + 轨迹(牵引车)”汇总保存到 CSV 文件（无注释头）。"""
        if planning_results is None or len(planning_results) == 0:
            print(" 规划结果为空，无法保存重规划速度与轨迹")
            return False

        try:
            header_cols = [
                'replan_idx', 't_global', 't_rel', 'k',
                'tractor_x', 'tractor_y', 'tractor_yaw'
            ]
            for i in range(self.num_trailers):
                header_cols += [f'trailer_{i}_x', f'trailer_{i}_y', f'trailer_{i}_yaw']
            header_cols += ['v', 'a']

            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            with open(filename, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header_cols)

                for replan_idx, result in enumerate(planning_results):
                    best = result.get('best_tentacle', None)
                    sp = result.get('speed_profile', None)
                    if best is None or sp is None:
                        continue

                    traj = best.get('tractor_trajectory', None)
                    if traj is None or len(traj) < 2:
                        continue

                    trailer_trajs = best.get('trailer_trajectories', [])

                    t_list = sp.get('t', [])
                    s_list = sp.get('s', [])
                    v_list = sp.get('v', [])
                    a = float(sp.get('a', 0.0))
                    if not t_list or not s_list or not v_list:
                        continue

                    n = min(len(t_list), len(s_list), len(v_list))
                    traj_np = np.array(traj, dtype=float)
                    path_s = self._compute_arc_length_xy(traj_np[:, :2])
                    t_global = float(result.get('time', 0.0))

                    for k in range(n):
                        t_rel = float(t_list[k])
                        s_k = float(s_list[k])
                        v_k = float(v_list[k])
                        i0, i1, alpha = self._find_segment_by_s(path_s, s_k)
                        p = self._interp_traj_by_index(traj, i0, i1, alpha)

                        row = [
                            int(replan_idx),
                            f"{t_global:.6f}",
                            f"{t_rel:.6f}",
                            int(k),
                            f"{float(p[0]):.6f}",
                            f"{float(p[1]):.6f}",
                            f"{float(p[2]):.6f}",
                        ]

                        for i in range(self.num_trailers):
                            if trailer_trajs is not None and i < len(trailer_trajs) and trailer_trajs[i] is not None and len(trailer_trajs[i]) > 0:
                                tp = self._interp_traj_by_index(trailer_trajs[i], i0, i1, alpha)
                                row += [
                                    f"{float(tp[0]):.6f}",
                                    f"{float(tp[1]):.6f}",
                                    f"{float(tp[2]):.6f}",
                                ]
                            else:
                                row += ["0.0", "0.0", "0.0"]

                        row += [f"{v_k:.6f}", f"{a:.6f}"]
                        writer.writerow(row)

            print(f" 重规划速度+轨迹 CSV 已保存到: {filename}")
            return True
        except Exception as e:
            print(f"[ERR] 保存重规划速度+轨迹 CSV 失败: {e}")
            return False


def main():
    """主函数"""
    class _Tee:
        def __init__(self, *streams):
            self._streams = [s for s in streams if s is not None]

        def write(self, data):
            for s in self._streams:
                try:
                    s.write(data)
                except Exception:
                    pass

        def flush(self):
            for s in self._streams:
                try:
                    s.flush()
                except Exception:
                    pass

    # 配置输入文件路径
    input_dir = r'D:\Javier\kunming airport\Lattice-Planner-main12.29\input\right_turn'
    output_root = r'D:\Javier\kunming airport\Lattice-Planner-main12.29\output\right_turn'

    # 输出分流：txt/fig/debug
    output_txt_dir = os.path.join(output_root, 'txt')
    output_fig_dir = os.path.join(output_root, 'fig')
    output_debug_dir = os.path.join(output_root, 'debug')
    os.makedirs(output_txt_dir, exist_ok=True)
    os.makedirs(output_fig_dir, exist_ok=True)
    os.makedirs(output_debug_dir, exist_ok=True)

    log_time = datetime.now().strftime("%m-%d_%H-%M")
    log_txt = os.path.join(output_txt_dir, f'terminal_output_{log_time}.txt')

    # 使用最新的文件
    trajectory_file = os.path.join(input_dir, 'trajectory_12-15.txt')
    environment_file = os.path.join(input_dir, 'environment_12-15.txt')
    # 可选：动态障碍物轨迹（CSV: step,time,x,y,yaw,velocity,length,width）
    dynamic_obstacle_file = os.path.join(input_dir, 'dynamic_obstacle_12-15.txt')


    # 创建局部规划器（可自定义配置）
    config = {
        'planning_horizon': 6.0,       # 前瞻距离
        'replan_hz': 30.0,
        'dt': 0.01,                    # 与全局规划器一致（10ms）
        'num_tentacles': 15,
        'lateral_deviation': 3.5,
        'safety_margin': 1.0,          # 安全余量

        # 纵向（速度）规划约束（你给定的值）
        'a_max': 0.6,
        'a_min': -1.5,
        'v_max': 5.5,
        'a_lat_max': 4.5,

        # 动画（避免重规划次数大时 GIF 帧数爆炸）
        'animation_fps': 30,
        'animation_frames_per_replan': 4,

        # 静态图可视化抽样次数（覆盖全过程；GIF 保留完整重规划避免抽搐）
        'save_max_replans': 100,
    }

    # 终端输出全量落盘（同时仍输出到控制台）
    with open(log_txt, 'w', encoding='utf-8') as _log_f:
        _orig_out, _orig_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(_orig_out, _log_f)
        sys.stderr = _Tee(_orig_err, _log_f)
        try:
            print(f"[log] 终端输出写入: {log_txt}")

            planner = TentacleLocalPlanner(config=config, num_trailers=8)

            if not planner.load_global_trajectory(trajectory_file):
                return
            if not planner.load_environment(environment_file):
                return

            if os.path.exists(dynamic_obstacle_file):
                planner.load_dynamic_obstacles(dynamic_obstacle_file)
            else:
                print(f"\n[info] 未找到动态障碍物文件，跳过: {dynamic_obstacle_file}")

            result = planner.plan()
            if not result:
                print("[ERR] planner.plan() 返回空结果")
                return

            planning_results, local_execution_path = result

            # 静态图：从全程重规划中“均匀抽样”保留 N 次（覆盖全过程）
            save_max_replans = int(config.get('save_max_replans', 0) or 0)
            planning_results_static = planning_results
            local_execution_path_static = local_execution_path
            if save_max_replans > 0 and len(planning_results) > save_max_replans:
                n_full = int(len(planning_results))
                n_keep = int(save_max_replans)

                sample_idx = np.linspace(0, n_full - 1, num=n_keep)
                sample_idx = [int(round(x)) for x in sample_idx]
                sample_idx = sorted(set(sample_idx))
                if len(sample_idx) < n_keep:
                    for i in range(n_full):
                        if i not in sample_idx:
                            sample_idx.append(i)
                            if len(sample_idx) >= n_keep:
                                break
                    sample_idx = sorted(sample_idx)
                if len(sample_idx) > n_keep:
                    sample_idx = sample_idx[:n_keep]

                default_exec_step = max(
                    1,
                    int(
                        round(
                            float(config.get('replan_interval', 1.0)) / float(config.get('dt', 0.01))
                        )
                    ),
                )
                cum_points = [1]
                for r in planning_results:
                    exec_step = int(r.get('execution_step', default_exec_step))
                    exec_step = max(1, exec_step)
                    add_n = int((exec_step + 1) // 2)
                    cum_points.append(cum_points[-1] + add_n)

                planning_results_static = []
                for i in sample_idx:
                    r = planning_results[i]
                    r_vis = dict(r)
                    r_vis['orig_replan_index'] = int(i)
                    r_vis['exec_path_start_count'] = int(cum_points[i])
                    r_vis['exec_path_end_count'] = int(cum_points[i + 1])
                    planning_results_static.append(r_vis)

                local_execution_path_static = local_execution_path

                print(
                    f"\n[info] 静态图从 {n_full} 次重规划中均匀抽样 {len(planning_results_static)} 次"
                    f"（执行路径点: {len(local_execution_path_static)}）"
                )

            current_time = datetime.now().strftime("%m-%d_%H-%M")

            local_path_csv = os.path.join(output_txt_dir, f'local_execution_path_{current_time}.csv')
            planner.save_local_path_to_csv(local_execution_path, local_path_csv)

            complete_path_csv = os.path.join(output_txt_dir, f'complete_trajectory_{current_time}.csv')
            planner.save_complete_trajectory_with_trailers_to_csv(planning_results, complete_path_csv)

            static_path = os.path.join(output_fig_dir, f'tentacle_local_planning_{current_time}.png')
            planner.visualize_planning_result(
                planning_results_static,
                local_execution_path_static,
                save_path=static_path,
                show_all_tentacles=False,
            )

            gif_path = os.path.join(output_fig_dir, f'tentacle_animation_{current_time}.gif')
            planner.create_animation(
                planning_results,
                local_execution_path,
                save_path=gif_path,
            )

            replans_speed_traj_csv = os.path.join(output_txt_dir, f'replans_speed_trajectory_{current_time}.csv')
            planner.save_all_replans_speed_and_trajectory_to_csv(planning_results, replans_speed_traj_csv)

            print(f"\n{'='*70}")
            print("[OK] 局部路径规划完成")
            print("    输出文件:")
            print(f"  - 局部执行路径: {local_path_csv}")
            print(f"  - 完整轨迹: {complete_path_csv}")
            print(f"  - 重规划速度+轨迹: {replans_speed_traj_csv}")
            print(f"  - 静态图: {static_path}")
            print(f"  - 动画: {gif_path}")
            print(f"  - 终端输出日志: {log_txt}")
            print(f"{'='*70}")

        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            sys.stdout, sys.stderr = _orig_out, _orig_err

    print(f"[log] 终端输出日志已保存: {log_txt}")


if __name__ == "__main__":
    main()
