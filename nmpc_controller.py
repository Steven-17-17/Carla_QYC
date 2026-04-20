import numpy as np
from scipy.optimize import minimize


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class NMPCController:
    def __init__(
        self,
        N=15,
        dt=0.1,
        wheelbase=3.0,
        x_offset_rear=0.0,
        x_offset_front=0.0,
        adaptive_horizon=True,
        N_min=8,
        N_max=22,
        curv_soft=0.015,
        curv_hard=0.09,
    ):
        """
        牵引车三状态 NMPC：优化 [v, steer]，挂车状态在控制外递推。
        """
        self.N = int(N)
        self.dt = float(dt)

        # 几何参数（按用户给定）
        self.L = float(wheelbase)
        self.x_offset_rear = float(x_offset_rear)
        self.x_offset_front = float(x_offset_front)

        # 约束参数（按用户要求：不倒车，30deg 转角，25km/h）
        self.v_min = 0.0
        self.v_max = 25.0 / 3.6
        self.steer_max = np.deg2rad(30.0)
        self.steer_rate_max = 0.5  # rad/s
        self.a_max = 2.8           # m/s^2
        self.a_min = -3.0          # m/s^2

        # 曲率自适应时域
        self.adaptive_horizon = bool(adaptive_horizon)
        self.N_min = int(min(N_min, N_max))
        self.N_max = int(max(N_min, N_max))
        self.curv_soft = float(curv_soft)
        self.curv_hard = float(max(curv_hard, curv_soft + 1e-6))

        # 代价权重
        self.Q_lat = 16.0
        self.Q_lon = 2.0
        self.Q_yaw = 4.0
        self.Q_v_ref = 4.0
        self.R_v = 0.05
        self.R_steer = 0.8
        self.R_acc = 0.15
        self.R_steer_rate = 2.0
        self.R_steer_ref = 1.2

        # 近端优先，抑制为了远端点而提前转弯
        self.stage_discount = 0.92

        # 放宽转角变化率硬约束，避免“必须提前打方向”
        self.steer_rate_max = 1.2

        self.last_v = 0.0
        self.last_steer = 0.0
        self.u0 = np.zeros(2 * self.N)

    def _estimate_local_curvature(self, target_traj):
        if target_traj.shape[0] < 3:
            return 0.0

        dyaw = np.array([normalize_angle(target_traj[i + 1, 2] - target_traj[i, 2]) for i in range(target_traj.shape[0] - 1)])
        ds = np.hypot(np.diff(target_traj[:, 0]), np.diff(target_traj[:, 1]))
        ds = np.maximum(ds, 1e-4)
        kappa = np.abs(dyaw) / ds
        if kappa.size == 0:
            return 0.0
        return float(np.median(kappa))

    def _compute_effective_horizon(self, target_traj):
        n_available = int(target_traj.shape[0])
        if not self.adaptive_horizon:
            return max(1, min(self.N, n_available))

        kappa = self._estimate_local_curvature(target_traj)
        if kappa <= self.curv_soft:
            n_eff = self.N_max
        elif kappa >= self.curv_hard:
            n_eff = self.N_min
        else:
            ratio = (kappa - self.curv_soft) / (self.curv_hard - self.curv_soft)
            n_eff = int(round(self.N_max - ratio * (self.N_max - self.N_min)))

        return max(1, min(n_eff, n_available))

    def _resize_warm_start(self, n_eff, current_v, current_steer):
        target_len = 2 * n_eff
        if self.u0.size == target_len:
            u = self.u0.copy()
        elif self.u0.size > target_len:
            u = self.u0[:target_len].copy()
        else:
            u = np.zeros(target_len)
            if self.u0.size > 0:
                u[:self.u0.size] = self.u0
                u[self.u0.size:] = self.u0[-1]
            else:
                u[0::2] = current_v
                u[1::2] = current_steer

        u[0::2] = np.clip(u[0::2], self.v_min, self.v_max)
        u[1::2] = np.clip(u[1::2], -self.steer_max, self.steer_max)
        return u

    def _build_rate_constraints(self, n_eff, v_prev, steer_prev):
        cons = []
        dv_up = self.a_max * self.dt
        dv_down = self.a_min * self.dt
        ds = self.steer_rate_max * self.dt

        for k in range(n_eff):
            i_v = 2 * k
            i_s = i_v + 1
            if k == 0:
                v_ref_idx = None
                s_ref_idx = None
                v_ref_val = float(v_prev)
                s_ref_val = float(steer_prev)
            else:
                v_ref_idx = 2 * (k - 1)
                s_ref_idx = v_ref_idx + 1
                v_ref_val = 0.0
                s_ref_val = 0.0

            if v_ref_idx is None:
                cons.append({'type': 'ineq', 'fun': lambda u, i=i_v, v0=v_ref_val: dv_up - (u[i] - v0)})
                cons.append({'type': 'ineq', 'fun': lambda u, i=i_v, v0=v_ref_val: (u[i] - v0) - dv_down})
            else:
                cons.append({'type': 'ineq', 'fun': lambda u, i=i_v, j=v_ref_idx: dv_up - (u[i] - u[j])})
                cons.append({'type': 'ineq', 'fun': lambda u, i=i_v, j=v_ref_idx: (u[i] - u[j]) - dv_down})

            if s_ref_idx is None:
                cons.append({'type': 'ineq', 'fun': lambda u, i=i_s, s0=s_ref_val: ds - (u[i] - s0)})
                cons.append({'type': 'ineq', 'fun': lambda u, i=i_s, s0=s_ref_val: (u[i] - s0) + ds})
            else:
                cons.append({'type': 'ineq', 'fun': lambda u, i=i_s, j=s_ref_idx: ds - (u[i] - u[j])})
                cons.append({'type': 'ineq', 'fun': lambda u, i=i_s, j=s_ref_idx: (u[i] - u[j]) + ds})

        return cons

    def _extract_state(self, current_state):
        state = np.asarray(current_state, dtype=float).reshape(-1)
        if state.size < 3:
            raise ValueError('current_state 至少需要 [x, y, yaw]')
        x, y, yaw = float(state[0]), float(state[1]), float(state[2])
        if state.size >= 4:
            current_v = float(state[3])
        else:
            current_v = float(self.last_v)
        return np.array([x, y, yaw], dtype=float), np.clip(current_v, self.v_min, self.v_max)

    def _ref_curvature_at(self, target_traj, k):
        n = target_traj.shape[0]
        if n < 2:
            return 0.0
        if k >= n - 1:
            k0 = max(0, n - 2)
            k1 = n - 1
        else:
            k0 = k
            k1 = k + 1

        dyaw = normalize_angle(target_traj[k1, 2] - target_traj[k0, 2])
        ds = float(np.hypot(target_traj[k1, 0] - target_traj[k0, 0], target_traj[k1, 1] - target_traj[k0, 1]))
        ds = max(ds, 1e-4)
        return float(dyaw / ds)

    def kinematic_bicycle_model(self, state, v, steer):
        """
        牵引车几何自行车模型。默认 state 为后轴参考点；若有偏移则先映射到后轴再积分。
        """
        x_ref, y_ref, yaw = state

        x_rear = x_ref - self.x_offset_rear * np.cos(yaw)
        y_rear = y_ref - self.x_offset_rear * np.sin(yaw)

        yaw_next = normalize_angle(yaw + (v / self.L) * np.tan(steer) * self.dt)
        x_rear_next = x_rear + v * np.cos(yaw) * self.dt
        y_rear_next = y_rear + v * np.sin(yaw) * self.dt

        x_ref_next = x_rear_next + self.x_offset_rear * np.cos(yaw_next)
        y_ref_next = y_rear_next + self.x_offset_rear * np.sin(yaw_next)

        return np.array([x_ref_next, y_ref_next, yaw_next], dtype=float)

    def cost_function(self, u, state_init, target_traj, target_speed):
        n_eff = int(target_traj.shape[0])
        state = state_init.copy()
        cost = 0.0

        for k in range(n_eff):
            v_k = u[2 * k]
            steer_k = u[2 * k + 1]

            state = self.kinematic_bicycle_model(state, v_k, steer_k)
            tx, ty, tyaw = target_traj[k]

            ex = state[0] - tx
            ey = state[1] - ty
            e_lat = -np.sin(tyaw) * ex + np.cos(tyaw) * ey
            e_lon = np.cos(tyaw) * ex + np.sin(tyaw) * ey
            yaw_err = normalize_angle(state[2] - tyaw)

            # 近端优先加权（k 越小权重越大）
            w_k = self.stage_discount ** k
            if k == n_eff - 1:
                w_k *= 2.5

            cost += w_k * (self.Q_lat * e_lat ** 2 + self.Q_lon * e_lon ** 2 + self.Q_yaw * yaw_err ** 2)

            cost += w_k * (self.Q_v_ref * (v_k - target_speed) ** 2)
            cost += w_k * (self.R_v * v_k ** 2)

            kappa_ref = self._ref_curvature_at(target_traj, k)
            steer_ref = np.arctan(self.L * kappa_ref)
            steer_ref = float(np.clip(steer_ref, -self.steer_max, self.steer_max))
            cost += w_k * (self.R_steer * steer_k ** 2 + self.R_steer_ref * (steer_k - steer_ref) ** 2)

            if k == 0:
                dv = (v_k - self.last_v) / self.dt
                ds = (steer_k - self.last_steer) / self.dt
            else:
                dv = (v_k - u[2 * (k - 1)]) / self.dt
                ds = (steer_k - u[2 * (k - 1) + 1]) / self.dt

            cost += w_k * (self.R_acc * dv ** 2 + self.R_steer_rate * ds ** 2)

        return float(cost)

    def solve(self, current_state, target_trajectory, target_speed=None):
        """
        求解 NMPC。

        Args:
            current_state: [x, y, yaw] 或 [x, y, yaw, v]
            target_trajectory: shape(n, 3)，其中列为 [x, y, yaw]
            target_speed: 速度参考（m/s），若不传则默认上一时刻速度
        Returns:
            v_opt, steer_opt
        """
        traj = np.asarray(target_trajectory, dtype=float)
        if traj.ndim != 2 or traj.shape[0] == 0 or traj.shape[1] < 3:
            return float(self.last_v), float(self.last_steer)

        state_init, current_v = self._extract_state(current_state)
        n_eff = self._compute_effective_horizon(traj)
        traj_eff = traj[:n_eff, :3]

        if target_speed is None:
            target_speed = current_v
        target_speed = float(np.clip(target_speed, self.v_min, self.v_max))

        u_guess = self._resize_warm_start(n_eff, current_v, self.last_steer)

        bounds = []
        for _ in range(n_eff):
            bounds.append((self.v_min, self.v_max))
            bounds.append((-self.steer_max, self.steer_max))

        constraints = self._build_rate_constraints(n_eff, current_v, self.last_steer)

        res = minimize(
            self.cost_function,
            u_guess,
            args=(state_init, traj_eff, target_speed),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 80, 'ftol': 1e-4, 'disp': False},
        )

        if res.success or res.status == 8:
            u_opt = res.x
        else:
            print('[NMPC Warning] 优化未收敛，回退 warm-start')
            u_opt = u_guess

        if u_opt.size >= 2:
            v_cmd = float(np.clip(u_opt[0], self.v_min, self.v_max))
            steer_cmd = float(np.clip(u_opt[1], -self.steer_max, self.steer_max))
        else:
            v_cmd = float(current_v)
            steer_cmd = float(self.last_steer)

        u_shift = np.roll(u_opt, -2)
        if u_shift.size >= 2:
            u_shift[-2:] = u_opt[-2:]
        self.u0 = u_shift
        self.last_v = v_cmd
        self.last_steer = steer_cmd

        return v_cmd, steer_cmd
