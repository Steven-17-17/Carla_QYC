import numpy as np
import math
from scipy.optimize import minimize

class NMPCController:
    def __init__(self, N=10, dt=0.1):
        """
        NMPC 控制器 (基于 scipy.optimize.minimize)
        
        Args:
            N (int): 预测地平线长度 (预测几步)
            dt (float): 离散步长 (秒)
        """
        self.N = N
        self.dt = dt
        
        # 牵引车物理约束
        self.v_min = -5.0   # [m/s]
        self.v_max = 10.0   # [m/s]
        self.steer_max = 50.0 * np.pi / 180.0  # 最大前轮转角 [rad]
        self.steer_rate_max = 0.5  # 方向盘最大转角率 [rad/s]
        
        # 权重权重
        self.Q_x = 10.0
        self.Q_y = 10.0
        self.Q_yaw = 5.0
        self.R_v = 1.0
        self.R_steer = 1.0
        self.R_steer_rate = 5.0
        
        # 初始化控制猜测 u = [v0, steer0, v1, steer1, ..., vN-1, steerN-1]
        self.u0 = np.zeros(2 * self.N)

    def kinematic_bicycle_model(self, state, v, steer):
        """
        牵引车几何自行车模型 (离散差分方程)
        """
        x, y, yaw = state
        L = 2.85  # 轴距 [m]，可按照林肯MKZ调整
        
        x_next = x + v * np.cos(yaw) * self.dt
        y_next = y + v * np.sin(yaw) * self.dt
        yaw_next = yaw + (v / L) * np.tan(steer) * self.dt
        
        # 角度归一化
        yaw_next = (yaw_next + np.pi) % (2 * np.pi) - np.pi
        
        return np.array([x_next, y_next, yaw_next])

    def cost_function(self, u, state_init, target_traj):
        """
        代价函数
        u: 2N 维向量，[v0, steer0, v1, steer1, ...]
        state_init: 初始状态 [x, y, yaw]
        target_traj: 目标轨迹 [N, 3] (x, y, yaw)
        """
        cost = 0.0
        state = state_init.copy()
        
        for k in range(self.N):
            v = u[2*k]
            steer = u[2*k + 1]
            
            # 使用模型步进一帧
            state = self.kinematic_bicycle_model(state, v, steer)
            
            # 位置与偏航角误差惩罚
            tx, ty, tyaw = target_traj[k]
            cost += self.Q_x * (state[0] - tx)**2
            cost += self.Q_y * (state[1] - ty)**2
            yaw_err = (state[2] - tyaw + np.pi) % (2*np.pi) - np.pi
            cost += self.Q_yaw * (yaw_err)**2
            
            # 控制努力惩罚
            cost += self.R_v * v**2
            cost += self.R_steer * steer**2
            
            # 控制平滑度惩罚 (转角率)
            if k > 0:
                prev_steer = u[2*(k-1) + 1]
                cost += self.R_steer_rate * ((steer - prev_steer) / self.dt)**2
                
        return cost

    def solve(self, current_state, target_trajectory):
        """
        求解NMPC
        Args:
            current_state (list/np.array): [x, y, yaw(rad)]
            target_trajectory (np.array): shape(N, 3) 包含 N 个参考点的 (x, y, yaw)
            
        Returns:
            v_opt, steer_opt: 当前步最优线速度 [m/s] 和 转角 [rad]
        """
        # 约束条件：速度和转角的绝对限制
        bounds = []
        for _ in range(self.N):
            bounds.append((self.v_min, self.v_max))
            bounds.append((-self.steer_max, self.steer_max))
            
        # 调用优化器 (SLSQP 算法替代 fmincon-SQP)
        res = minimize(
            self.cost_function,
            self.u0,
            args=(np.array(current_state), target_trajectory),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-3, 'disp': False}
        )
        
        if res.success or res.status == 8: # 8 is positive directional derivative (often acceptable enough)
            self.u0 = np.roll(res.x, -2)   # Shift memory left for warm-start
            self.u0[-2:] = res.x[-2:]      # Copy last control
            return res.x[0], res.x[1]
        else:
            print("[NMPC Warning] 优化未收敛")
            return self.u0[0], self.u0[1]
