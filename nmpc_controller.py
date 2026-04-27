"""
基于MATLAB NMPC核心思路的牵引车NMPC控制器
只针对单牵引车，不考虑挂车
"""
import numpy as np
from scipy.optimize import minimize


def normalize_angle(angle):
    """归一化角度到 [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def smooth_path_and_calculate_yaw(original_path, ds=0.1, smooth_factor=0.85):
    """
    路径平滑 + 自动计算连续的航向角
    兼容版本，不用 make_splrep
    original_path: 原始路径 Nx2 或 Nx3
    ds: 采样间隔 (米)
    smooth_factor: 平滑强度 (0-1, 越大越平滑)
    """
    original_path = np.asarray(original_path)
    
    if len(original_path) < 3:
        # 点太少，直接简单处理
        if original_path.shape[1] == 2:
            original_path = np.column_stack([original_path, np.zeros(len(original_path))])
        return original_path
    
    # ============= 1. 先对原始路径做移动平均平滑 =============
    smooth_window = max(3, int(smooth_factor * 7))
    if len(original_path) > smooth_window:
        x_orig = original_path[:, 0]
        y_orig = original_path[:, 1]
        
        x_smooth = np.zeros_like(x_orig)
        y_smooth = np.zeros_like(y_orig)
        
        half_win = smooth_window // 2
        
        x_smooth[:half_win] = x_orig[:half_win]
        y_smooth[:half_win] = y_orig[:half_win]
        x_smooth[-half_win:] = x_orig[-half_win:]
        y_smooth[-half_win:] = y_orig[-half_win:]
        
        for i in range(half_win, len(x_orig) - half_win):
            x_smooth[i] = np.mean(x_orig[i-half_win:i+half_win+1])
            y_smooth[i] = np.mean(y_orig[i-half_win:i+half_win+1])
        
        smoothed_original = np.column_stack([x_smooth, y_smooth])
    else:
        smoothed_original = original_path[:, :2]
    
    # ============= 2. 计算累计弧长 =============
    s = [0.0]
    for i in range(1, len(smoothed_original)):
        dx = smoothed_original[i, 0] - smoothed_original[i-1, 0]
        dy = smoothed_original[i, 1] - smoothed_original[i-1, 1]
        s.append(s[-1] + np.hypot(dx, dy))
    
    s = np.array(s)
    
    # 检查并处理严格递增
    # 合并过近的点
    keep_indices = [0]
    for i in range(1, len(s)):
        if s[i] - s[keep_indices[-1]] > 1e-3:  # 至少0.001m间隔
            keep_indices.append(i)
    
    s_clean = s[keep_indices]
    xy_clean = smoothed_original[keep_indices, :]
    
    if len(s_clean) < 2:
        # 数据太少，直接返回简单版本
        if original_path.shape[1] == 2:
            original_path = np.column_stack([original_path, np.zeros(len(original_path))])
        return original_path
    
    total_length = s_clean[-1]
    
    # ============= 3. 用 CubicSpline 插值 (兼容) =============
    from scipy.interpolate import CubicSpline
    
    cs_x = CubicSpline(s_clean, xy_clean[:, 0], bc_type='natural')
    cs_y = CubicSpline(s_clean, xy_clean[:, 1], bc_type='natural')
    
    # 重新采样
    num_points = int(np.ceil(total_length / ds)) + 1
    s_new = np.linspace(0, total_length, num_points)
    x_new = cs_x(s_new)
    y_new = cs_y(s_new)
    
    # ============= 4. 从位置计算航向角 =============
    # 计算切线方向
    dx = np.zeros_like(x_new)
    dy = np.zeros_like(y_new)
    
    # 边界
    dx[0] = x_new[1] - x_new[0]
    dy[0] = y_new[1] - y_new[0]
    dx[-1] = x_new[-1] - x_new[-2]
    dy[-1] = y_new[-1] - y_new[-2]
    
    # 中点
    dx[1:-1] = x_new[2:] - x_new[:-2]
    dy[1:-1] = y_new[2:] - y_new[:-2]
    
    yaw_new = np.arctan2(dy, dx)
    
    # ============= 5. 平滑航向角 =============
    window_size = 7
    if len(yaw_new) > window_size:
        yaw_unwrap = np.unwrap(yaw_new)
        yaw_smooth = np.convolve(yaw_unwrap, np.ones(window_size)/window_size, mode='same')
        yaw_new = yaw_smooth
    
    # 归一化
    yaw_new = normalize_angle(yaw_new)
    
    return np.column_stack([x_new, y_new, yaw_new])


def interpolate_path(original_path, ds=0.1):
    """
    旧的函数保留，内部调用新的平滑器
    """
    return smooth_path_and_calculate_yaw(original_path, ds=ds)


class NMPCController:
    def __init__(
        self,
        N=30,
        dt=0.05,
        wheelbase=4.7,
    ):
        """
        初始化NMPC控制器
        """
        self.N_p = N
        self.dt = dt
        self.l1 = wheelbase  # 牵引车轴距
        
        # 控制约束
        self.max_steer = np.deg2rad(30.0)
        self.min_v = 0.0
        self.max_v = 3.0  # 最大速度 3 m/s
        
        # 热启动用的上一帧解
        self.last_u_seq = None
        
    def predict(self, xt, u_seq):
        """
        预测轨迹
        xt: 当前状态 [x, y, yaw, v]
        u_seq: 控制序列 [ [delta, v], ... ]
        """
        x_pred = [xt.copy()]
        for delta, v_cmd in u_seq:
            x_curr = x_pred[-1].copy()
            x_curr[3] = v_cmd
            
            # 单车运动学模型
            dx = np.zeros(4)
            dx[0] = x_curr[3] * np.cos(x_curr[2])
            dx[1] = x_curr[3] * np.sin(x_curr[2])
            dx[2] = x_curr[3] * np.tan(delta) / self.l1
            dx[3] = 0.0  # 速度由v_cmd决定
            
            x_next = x_curr + dx * self.dt
            x_next[2] = normalize_angle(x_next[2])
            x_pred.append(x_next)
        
        return np.array(x_pred)
    
    def cost_function(self, u_vec, x0, ref_trajectory, v_target=1.0):
        """
        NMPC代价函数
        """
        u_seq = u_vec.reshape(self.N_p, 2)  # [delta, v]
        
        J = 0.0
        
        # 初始化预测
        xt = x0.copy()
        
        # 找当前最近参考点
        P_f0_x = xt[0] + self.l1 * np.cos(xt[2])
        P_f0_y = xt[1] + self.l1 * np.sin(xt[2])
        dist_sq = (ref_trajectory[:, 0] - P_f0_x)**2 + (ref_trajectory[:, 1] - P_f0_y)**2
        current_idx = np.argmin(dist_sq)
        
        # 假设参考路径已经是等距采样
        ds = 0.1  # 参考点间距
        accumulated_s = 0.0
        
        for i in range(self.N_p):
            delta_cmd = u_seq[i, 0]
            v_cmd = u_seq[i, 1]
            
            # 惩罚速度过小或为负
            if v_cmd < 0.1:
                J += 1000.0 * (0.1 - v_cmd)**2
            
            # 临时替换速度
            xt[3] = v_cmd
            
            # 运动学积分
            dx = np.zeros(4)
            dx[0] = xt[3] * np.cos(xt[2])
            dx[1] = xt[3] * np.sin(xt[2])
            dx[2] = xt[3] * np.tan(delta_cmd) / self.l1
            dx[3] = 0.0
            
            xt = xt + dx * self.dt
            xt[2] = normalize_angle(xt[2])
            
            # 基于累计弧长匹配参考点
            accumulated_s = accumulated_s + v_cmd * self.dt
            idx_offset = int(np.round(accumulated_s / ds))
            target_idx = current_idx + idx_offset
            target_idx = min(target_idx, ref_trajectory.shape[0] - 1)
            ref_p = ref_trajectory[target_idx, :]
            
            # 计算前轴位置
            P_f = np.array([
                xt[0] + self.l1 * np.cos(xt[2]),
                xt[1] + self.l1 * np.sin(xt[2])
            ])
            
            # 横向误差
            dx_err = P_f[0] - ref_p[0]
            dy_err = P_f[1] - ref_p[1]
            cte = dx_err * (-np.sin(ref_p[2])) + dy_err * np.cos(ref_p[2])
            
            # 航向误差
            yaw_err = normalize_angle(xt[2] - ref_p[2])
            
            # 权重设计
            W_cte = 10.0
            W_yaw = 20.0
            
            J = J + W_cte * cte**2 + W_yaw * yaw_err**2
            
            # 速度跟踪（优先用触须给的）
            W_v = 50.0
            J = J + W_v * (v_cmd - v_target)**2
            
            # 控制量惩罚
            W_du_steer = 30.0
            W_du_v = 10.0
            
            if i > 0:
                J = J + W_du_steer * (u_seq[i, 0] - u_seq[i-1, 0])**2
                J = J + W_du_v * (u_seq[i, 1] - u_seq[i-1, 1])**2
            else:
                J = J + 50.0 * u_seq[i, 0]**2
        
        return J
    
    def solve(self, current_state, target_trajectory, target_speed=None):
        """
        求解NMPC优化
        返回: throttle, brake, steer (-1到1)
        """
        # 状态: x, y, yaw, v
        x0 = np.array([
            current_state[0],
            current_state[1],
            current_state[2],
            current_state[3] if len(current_state) > 3 else 1.0
        ])
        
        ref_traj = np.asarray(target_trajectory)
        
        # 平滑路径 + 计算连续航向角 + 0.1m采样
        ref_traj = smooth_path_and_calculate_yaw(ref_traj, ds=0.1, smooth_factor=0.85)
        
        # 目标速度（优先用触须给的）
        if target_speed is not None:
            v_target = max(target_speed, 0.5)
        else:
            v_target = max(x0[3], 0.5)
        
        # 热启动
        if self.last_u_seq is None:
            self.last_u_seq = np.zeros((self.N_p, 2))
            self.last_u_seq[:, 1] = v_target
        
        u0_seq = np.vstack([self.last_u_seq[1:, :], self.last_u_seq[-1, :]])
        u0_vec = u0_seq.flatten()
        
        # 边界约束
        lb = []
        ub = []
        for _ in range(self.N_p):
            lb.extend([-self.max_steer, max(self.min_v, 0.1)])  # 最小速度至少0.1
            ub.extend([self.max_steer, self.max_v])
        
        # 优化求解
        try:
            result = minimize(
                fun=self.cost_function,
                x0=u0_vec,
                args=(x0, ref_traj, v_target),
                bounds=list(zip(lb, ub)),
                method='SLSQP',
                options={
                    'maxiter': 50,
                    'ftol': 1e-4,
                    'disp': False
                }
            )
            
            if result.success:
                u_seq_mat = result.x.reshape(self.N_p, 2)
                u_opt = u_seq_mat[0, :]
                self.last_u_seq = u_seq_mat
            else:
                u_opt = self.last_u_seq[1, :]
        
        except Exception as e:
            u_opt = self.last_u_seq[1, :]
        
        delta_cmd, v_cmd = u_opt
        
        # ===================== 转换为Carla的控制量 =====================
        # 1. 转向角
        steer_out = delta_cmd / self.max_steer
        steer_out = np.clip(steer_out, -1.0, 1.0)
        
        # 2. 速度转换为油门/刹车
        current_v = x0[3]
        # 直接用触须给的速度作为目标
        v_target_use = v_target if target_speed is not None else max(v_cmd, 0.5)
        speed_error = v_target_use - current_v
        
        if v_target_use < 0.3:
            throttle = 0.0
            brake = 1.0
        else:
            if speed_error > 0.0:
                throttle = min(1.0, 0.3 + speed_error * 2.0)
                brake = 0.0
                
                if current_v < 1.0 and v_target_use > 1.5:
                    throttle = max(throttle, 0.7)
            else:
                throttle = 0.0
                brake = min(1.0, -speed_error * 0.8)
        
        return throttle, brake, steer_out
