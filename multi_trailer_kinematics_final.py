"""
多节全挂车运动学模型库

这个库提供了多节全挂车的运动学模型，用于计算车辆在指定控制输入下的下一时间步长状态。

主要类：
- MultiTrailerKinematics: 多节全挂车运动学模型的主类
- VehicleParams: 车辆参数配置类
"""

import math
import numpy as np

# 铰链游隙相关常数
HITCH_RING_RADIUS_M = 0.04
HITCH_PIN_RADIUS_M = 0.02
HITCH_HYSTERESIS_RATIO = 1.5
HITCH_RATE_LIMIT_RAD_S = 1.0


def normalize_angle(angle):
    """
    将角度归一化到 [-π, π] 范围内
    
    Args:
        angle: 输入角度（弧度）
        
    Returns:
        归一化后的角度
    """
    a = math.fmod(angle + np.pi, 2 * np.pi)
    if a < 0.0:
        a += (2.0 * np.pi)
    return a - np.pi


def _hitch_clearance_m():
    """计算铰链游隙"""
    return max(0.0, float(HITCH_RING_RADIUS_M) - float(HITCH_PIN_RADIUS_M))


def _angle_deadband_for_clearance(clearance_m, lever_m):
    """根据游隙和杠杆长度计算角度死区"""
    lever = max(float(lever_m), 1e-6)
    return float(math.atan2(float(clearance_m), lever))


def _apply_hysteresis(prev_angle, desired_angle, enter_db, exit_db):
    """应用滞后效应"""
    delta = normalize_angle(desired_angle - prev_angle)
    ad = abs(delta)
    if ad <= enter_db:
        return prev_angle
    if ad >= exit_db:
        return desired_angle
    return prev_angle


def _apply_rate_limit(prev_angle, target_angle, max_rate, dt):
    """应用速率限制"""
    max_delta = float(max_rate) * float(dt)
    delta = normalize_angle(target_angle - prev_angle)
    if abs(delta) <= max_delta:
        return target_angle
    return normalize_angle(prev_angle + math.copysign(max_delta, delta))



class VehicleParams:
    """
    车辆参数配置类
    
    Attributes:
        num_trailers: 挂车节数
        L: 牵引车轴距
        La: 牵引车铰接点到前轴中心的距离
        Lb: 牵引车铰接点到后轴中心的距离
        trailer_L: 各挂车轴距列表
        trailer_Lb: 各挂车后轴中心到铰接点距离列表
        max_steer: 最大前轮转角
    """
    
    def __init__(self, num_trailers=1, L=3.0, La=2.7, Lb=0.8, max_steer=0.6):
        """
        初始化车辆参数
        
        Args:
            num_trailers: 挂车节数，默认为1 (支持最多8节)
            L: 牵引车轴距，默认为3.0
            La: 铰接点到前轴中心的距离，默认为2.7
            Lb: 铰接点到后轴中心的距离，默认为0.8
            max_steer: 最大前轮转角，默认为0.6
        """
        self.num_trailers = num_trailers
        self.L = L
        self.La = La
        self.Lb = Lb
        self.max_steer = max_steer
        
        # 初始化挂车参数（支持8节全挃车）
        # 配置规则：与 duojie11.11.py 保持一致
        # - 奇数节(1,3,5,7...): trailer_L=1.0, trailer_Lb=0
        # - 偶数节(2,4,6,8...): trailer_L=4.0, trailer_Lb=0.8
        self.trailer_L = []
        self.trailer_Lb = []
        
        for i in range(num_trailers):
            if (i + 1) % 2 == 1:  # 奇数节 (1, 3, 5, 7, ...)
                self.trailer_L.append(1.4)     # 奇数节挂车轴距
                self.trailer_Lb.append(0)      # 奇数节挂车后轴中心到铰接点距离
            else:  # 偶数节 (2, 4, 6, 8, ...)
                self.trailer_L.append(2.0)     # 偶数节挂车轴距
                self.trailer_Lb.append(0.5)    # 偶数节挂车后轴中心到铰接点距离


class MultiTrailerKinematics:
    """
    多节全挂车运动学模型
    
    该类实现了多节全挂车的运动学模型，可以根据牵引车的速度和转向角计算
    牵引车和各节挂车在下一时间步长的位置和航向角。
    
    Example:
        >>> params = VehicleParams(num_trailers=2)
        >>> model = MultiTrailerKinematics(params, dt=0.01)
        >>> 
        >>> # 初始状态：牵引车位置和航向角
        >>> tractor_state = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        >>> # 各节挂车的位置和航向角
        >>> trailer_states = [
        ...     {'x': -3.0, 'y': 0.0, 'yaw': 0.0},
        ...     {'x': -7.0, 'y': 0.0, 'yaw': 0.0}
        ... ]
        >>> 
        >>> # 控制输入：速度 2.5 m/s，转向角 0.3 rad
        >>> v = 2.5  # m/s
        >>> delta = 0.3  # rad
        >>> 
        >>> # 计算下一时间步长的状态
        >>> new_tractor, new_trailers = model.update_state(
        ...     tractor_state, trailer_states, v, delta
        ... )
    """
    
    def __init__(self, params, dt=0.1):
        """
        初始化多节全挂车运动学模型
        
        Args:
            params: VehicleParams 对象，包含车辆参数
            dt: 时间步长，默认为 0.1 秒
        """
        self.params = params
        self.dt = dt
    
    def initialize_trailers(self, tractor_state):
        """
        根据牵引车状态初始化所有挂车的状态。
        
        该方法模仿 duojie11.11.py 中的初始化逻辑，
        计算每节挂车的位置（包括后轴中心和铰接点）和航向角。
        
        初始化原理：
        - 第一节挂车：
          * 铰接点位置（前轴中心）在牵引车后轴中心沿牵引车航向后方 Lb 距离处
          * 后轴中心在铰接点沿该航向后方 trailer_L[0] 距离处
        - 第 i 节挂车：
          * 铰接点位置在第 i-1 节挂车后轴中心沿其航向后方 trailer_Lb[i-1] 距离处
          * 后轴中心在铰接点沿该航向后方 trailer_L[i] 距离处
        
        Args:
            tractor_state: 牵引车状态字典，包含:
                - 'x': 牵引车后轴中心 x 坐标
                - 'y': 牵引车后轴中心 y 坐标
                - 'yaw': 牵引车航向角（弧度）
        
        Returns:
            list: 各节挂车初始化状态，每个元素为字典，包含:
                - 'x': 后轴中心 x 坐标
                - 'y': 后轴中心 y 坐标
                - 'yaw': 航向角（初始时与参考物体相同）
                - 'x_front': 铰接点 x 坐标（前轴中心）
                - 'y_front': 铰接点 y 坐标（前轴中心）
        
        Example:
            >>> params = VehicleParams(num_trailers=2)
            >>> model = MultiTrailerKinematics(params, dt=0.01)
            >>> tractor = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
            >>> trailers = model.initialize_trailers(tractor)
            >>> print(len(trailers))  # 2
            >>> print(trailers[0])    # 第一节挂车状态
        """
        trailers = []
        
        for i in range(self.params.num_trailers):
            if i == 0:
                # 第一节挂车连接到牵引车
                # 计算铰接点（前轴中心）位置
                trailer_x_front = tractor_state['x'] - self.params.Lb * math.cos(tractor_state['yaw'])
                trailer_y_front = tractor_state['y'] - self.params.Lb * math.sin(tractor_state['yaw'])
                
                # 计算后轴中心位置
                trailer_x = trailer_x_front - self.params.trailer_L[i] * math.cos(tractor_state['yaw'])
                trailer_y = trailer_y_front - self.params.trailer_L[i] * math.sin(tractor_state['yaw'])
                trailer_yaw = tractor_state['yaw']
            else:
                # 后续挂车连接到前一节挂车
                prev_state = trailers[i - 1]
                prev_trailer_Lb = self.params.trailer_Lb[i - 1]
                
                # 计算铰接点（前轴中心）位置
                trailer_x_front = prev_state['x'] - prev_trailer_Lb * math.cos(prev_state['yaw'])
                trailer_y_front = prev_state['y'] - prev_trailer_Lb * math.sin(prev_state['yaw'])
                
                # 计算后轴中心位置
                trailer_x = trailer_x_front - self.params.trailer_L[i] * math.cos(prev_state['yaw'])
                trailer_y = trailer_y_front - self.params.trailer_L[i] * math.sin(prev_state['yaw'])
                trailer_yaw = prev_state['yaw']
            
            # 创建挂车状态字典
            trailer = {
                'x': trailer_x,
                'y': trailer_y,
                'yaw': trailer_yaw,
                'x_front': trailer_x_front,
                'y_front': trailer_y_front
            }
            trailers.append(trailer)
        
        return trailers
    
    def update_state(self, tractor_state, trailer_states, v, delta):
        """
        计算牵引车和挂车在下一时间步长的状态
        
        Args:
            tractor_state: 字典，包含牵引车当前状态
                - 'x': 牵引车后轴中心 x 坐标
                - 'y': 牵引车后轴中心 y 坐标
                - 'yaw': 牵引车航向角（弧度）
            trailer_states: 列表，包含各节挂车当前状态（字典列表）
                - 每个字典包含 'x', 'y', 'yaw' 字段
            v: 牵引车速度 (m/s)
            delta: 牵引车前轮转角 (弧度)
            
        Returns:
            tuple: (new_tractor_state, new_trailer_states)
                - new_tractor_state: 牵引车下一时间步长的状态
                - new_trailer_states: 各节挂车下一时间步长的状态列表
        """
        # 限制转向角
        delta = np.clip(delta, -self.params.max_steer, self.params.max_steer)
        
        # 复制状态以避免修改原始数据
        new_tractor_state = tractor_state.copy()
        new_trailer_states = [ts.copy() for ts in trailer_states]
        
        # 更新牵引车
        new_tractor_state['yaw'] += v / self.params.L * math.tan(delta) * self.dt
        new_tractor_state['yaw'] = normalize_angle(new_tractor_state['yaw'])
        
        new_tractor_state['x'] += v * math.cos(new_tractor_state['yaw']) * self.dt
        new_tractor_state['y'] += v * math.sin(new_tractor_state['yaw']) * self.dt
        
        # 更新每节挂车
        trailer_velocities = []
        for i in range(self.params.num_trailers):
            if i == 0:
                # 第一节挂车连接到牵引车
                trailer_v = self._update_first_trailer(
                    new_tractor_state, new_trailer_states[i], 
                    v, delta
                )
                trailer_velocities.append(trailer_v)
            else:
                # 后续挂车连接到前一节挂车
                trailer_v = self._update_subsequent_trailer(
                    new_trailer_states[i-1], new_trailer_states[i], 
                    i, trailer_velocities[i-1]
                )
                trailer_velocities.append(trailer_v)
        
        return new_tractor_state, new_trailer_states
    
    def _update_first_trailer(self, tractor_state, trailer_state, v, delta):
        """
        更新第一节挂车状态（包含游隙死区和位置校正）
        
        Args:
            tractor_state: 牵引车状态
            trailer_state: 第一节挂车状态（会被修改）
            v: 牵引车速度
            delta: 牵引车前轮转角
            
        Returns:
            trailer_v: 第一节挃车的速度（用于后续挃车更新）
        """
        # 计算铰接角
        hitch_angle = normalize_angle(tractor_state['yaw'] - trailer_state['yaw'])
        
        # 更新角度（yaw）
        dyaw = v / self.params.trailer_L[0] * (
            math.sin(hitch_angle) - 
            self.params.Lb * math.tan(delta) * 
            math.cos(hitch_angle) / self.params.L
        ) * self.dt
        
        trailer_state['yaw'] += dyaw
        trailer_state['yaw'] = normalize_angle(trailer_state['yaw'])
        
        # 计算速度
        trailer_v = v * (math.cos(hitch_angle) + self.params.Lb * math.tan(delta) * math.sin(hitch_angle) / self.params.L)
        
        # 前轴中心位置（铰接点）
        hitch_x = tractor_state['x'] - self.params.Lb * math.cos(tractor_state['yaw'])
        hitch_y = tractor_state['y'] - self.params.Lb * math.sin(tractor_state['yaw'])
        
        trailer_state['x_front'] = hitch_x
        trailer_state['y_front'] = hitch_y
        
        # 【改进】加入游隙死区逻辑
        hitch_error = normalize_angle(tractor_state['yaw'] - (trailer_state['yaw'] - dyaw))
        deadband = _angle_deadband_for_clearance(_hitch_clearance_m(), self.params.trailer_L[0])
        if abs(hitch_error) < deadband:
            # 游隙未被拉紧，挂车保持原来的角度（撤销转向）
            trailer_state['yaw'] -= dyaw
        
        # 【改进】强制后轴沿自身heading方向运动，避免漂移
        trailer_state['x'] += trailer_v * math.cos(trailer_state['yaw']) * self.dt
        trailer_state['y'] += trailer_v * math.sin(trailer_state['yaw']) * self.dt
        
        # 【改进】位置校正：确保几何约束（铰接点距离固定）
        dx = trailer_state['x_front'] - trailer_state['x']
        dy = trailer_state['y_front'] - trailer_state['y']
        dist = math.hypot(dx, dy)
        if dist > 0:
            correction = dist - self.params.trailer_L[0]
            trailer_state['x'] += correction * (dx / dist)
            trailer_state['y'] += correction * (dy / dist)
            trailer_state['yaw'] = math.atan2(dy, dx)
        
        return trailer_v
    
    def _update_subsequent_trailer(self, prev_trailer_state, curr_trailer_state, trailer_index, prev_trailer_v):
        """
        更新后续挂车状态（包含游隙死区和位置校正）
        
        Args:
            prev_trailer_state: 前一节挂车状态
            curr_trailer_state: 当前挂车状态（会被修改）
            trailer_index: 当前挂车索引（从1开始）
            prev_trailer_v: 前一节挃车的速度（用于计算当前挃车速度）
            
        Returns:
            curr_trailer_v: 当前挃车的速度（用于下一节挃车更新）
        """
        # 获取前一节和当前节的参数
        prev_trailer_L = self.params.trailer_L[trailer_index - 1]
        prev_trailer_Lb = self.params.trailer_Lb[trailer_index - 1]
        curr_trailer_L = self.params.trailer_L[trailer_index]
        
        # 计算铰接角（前一节到当前节的角度差）
        hitch_angle = normalize_angle(prev_trailer_state['yaw'] - curr_trailer_state['yaw'])
        
        # 更新航向角
        dyaw = prev_trailer_v / curr_trailer_L * (
            math.sin(hitch_angle) - 
            prev_trailer_Lb * math.tan(hitch_angle) * 
            math.cos(hitch_angle) / prev_trailer_L
        ) * self.dt
        
        curr_trailer_state['yaw'] += dyaw
        curr_trailer_state['yaw'] = normalize_angle(curr_trailer_state['yaw'])
        
        # 更新速度
        curr_trailer_v = prev_trailer_v * (math.cos(hitch_angle) + prev_trailer_Lb * math.tan(hitch_angle) * math.sin(hitch_angle) / prev_trailer_L)
        
        # 铰接点（前一节的前轴）位置
        hitch_x = prev_trailer_state['x'] - prev_trailer_Lb * math.cos(prev_trailer_state['yaw'])
        hitch_y = prev_trailer_state['y'] - prev_trailer_Lb * math.sin(prev_trailer_state['yaw'])
        
        curr_trailer_state['x_front'] = hitch_x
        curr_trailer_state['y_front'] = hitch_y
        
        # 【改进】加入游隙死区逻辑
        hitch_error = normalize_angle(prev_trailer_state['yaw'] - (curr_trailer_state['yaw'] - dyaw))
        deadband = _angle_deadband_for_clearance(_hitch_clearance_m(), curr_trailer_L)
        if abs(hitch_error) < deadband:
            # 游隙未被拉紧，挂车保持原来的角度（撤销转向）
            curr_trailer_state['yaw'] -= dyaw
        
        # 【改进】强制后轴沿自身heading方向运动，避免漂移
        curr_trailer_state['x'] += curr_trailer_v * math.cos(curr_trailer_state['yaw']) * self.dt
        curr_trailer_state['y'] += curr_trailer_v * math.sin(curr_trailer_state['yaw']) * self.dt
        
        # 【改进】位置校正：确保几何约束（铰接点距离固定）
        dx = curr_trailer_state['x_front'] - curr_trailer_state['x']
        dy = curr_trailer_state['y_front'] - curr_trailer_state['y']
        dist = math.hypot(dx, dy)
        if dist > 0:
            correction = dist - curr_trailer_L
            curr_trailer_state['x'] += correction * (dx / dist)
            curr_trailer_state['y'] += correction * (dy / dist)
            curr_trailer_state['yaw'] = math.atan2(dy, dx)
        
        return curr_trailer_v
    
    def update_state_with_velocities(self, tractor_state, trailer_states_with_velocities, v, delta):
        """
        计算牵引车和挂车在下一时间步长的状态（追踪各节的速度）
        
        这是一个改进的版本，可以正确处理各节挂车的速度计算。
        
        Args:
            tractor_state: 牵引车当前状态
                - 'x', 'y', 'yaw': 位置和航向角
            trailer_states_with_velocities: 各节挂车当前状态列表
                - 每个字典包含 'x', 'y', 'yaw', 'v', 'hitch_angle' 字段
            v: 牵引车速度 (m/s)
            delta: 牵引车前轮转角 (弧度)
            
        Returns:
            tuple: (new_tractor_state, new_trailer_states_with_velocities)
        """
        # 限制转向角
        delta = np.clip(delta, -self.params.max_steer, self.params.max_steer)
        
        # 复制状态
        new_tractor_state = tractor_state.copy()
        new_trailer_states = [ts.copy() for ts in trailer_states_with_velocities]
        
        # 更新牵引车
        new_tractor_state['yaw'] += v / self.params.L * math.tan(delta) * self.dt
        new_tractor_state['yaw'] = normalize_angle(new_tractor_state['yaw'])
        
        new_tractor_state['x'] += v * math.cos(new_tractor_state['yaw']) * self.dt
        new_tractor_state['y'] += v * math.sin(new_tractor_state['yaw']) * self.dt
        
        # 更新第一节挂车
        hitch_angle = normalize_angle(new_tractor_state['yaw'] - new_trailer_states[0]['yaw'])
        
        dyaw = v / self.params.trailer_L[0] * (
            math.sin(hitch_angle) - 
            self.params.Lb * math.tan(delta) * 
            math.cos(hitch_angle) / self.params.L
        ) * self.dt
        
        new_trailer_states[0]['yaw'] += dyaw
        new_trailer_states[0]['yaw'] = normalize_angle(new_trailer_states[0]['yaw'])
        
        new_trailer_states[0]['v'] = v * (
            math.cos(hitch_angle) + 
            self.params.Lb * math.tan(delta) * math.sin(hitch_angle) / self.params.L
        )
        new_trailer_states[0]['hitch_angle'] = normalize_angle(new_tractor_state['yaw'] - new_trailer_states[0]['yaw'])
        
        # 更新位置
        hitch_x = new_tractor_state['x'] - self.params.Lb * math.cos(new_tractor_state['yaw'])
        hitch_y = new_tractor_state['y'] - self.params.Lb * math.sin(new_tractor_state['yaw'])
        
        new_trailer_states[0]['x_front'] = hitch_x
        new_trailer_states[0]['y_front'] = hitch_y
        
        # 【改进】加入游隙死区逻辑
        hitch_error = normalize_angle(new_tractor_state['yaw'] - (new_trailer_states[0]['yaw'] - dyaw))
        deadband = _angle_deadband_for_clearance(_hitch_clearance_m(), self.params.trailer_L[0])
        if abs(hitch_error) < deadband:
            new_trailer_states[0]['yaw'] -= dyaw
        
        # 【改进】强制后轴沿自身heading方向运动
        new_trailer_states[0]['x'] += new_trailer_states[0]['v'] * math.cos(new_trailer_states[0]['yaw']) * self.dt
        new_trailer_states[0]['y'] += new_trailer_states[0]['v'] * math.sin(new_trailer_states[0]['yaw']) * self.dt
        
        # 【改进】位置校正
        dx = new_trailer_states[0]['x_front'] - new_trailer_states[0]['x']
        dy = new_trailer_states[0]['y_front'] - new_trailer_states[0]['y']
        dist = math.hypot(dx, dy)
        if dist > 0:
            correction = dist - self.params.trailer_L[0]
            new_trailer_states[0]['x'] += correction * (dx / dist)
            new_trailer_states[0]['y'] += correction * (dy / dist)
            new_trailer_states[0]['yaw'] = math.atan2(dy, dx)
        
        # 更新后续挂车
        for i in range(1, self.params.num_trailers):
            prev_state = new_trailer_states[i-1]
            curr_state = new_trailer_states[i]
            
            hitch_angle = normalize_angle(prev_state['yaw'] - curr_state['yaw'])
            
            dyaw = prev_state['v'] / self.params.trailer_L[i] * (
                math.sin(hitch_angle) - 
                self.params.trailer_Lb[i-1] * math.tan(hitch_angle) * 
                math.cos(hitch_angle) / self.params.trailer_L[i-1]
            ) * self.dt
            
            curr_state['yaw'] += dyaw
            curr_state['yaw'] = normalize_angle(curr_state['yaw'])
            
            curr_state['v'] = prev_state['v'] * (
                math.cos(hitch_angle) + 
                self.params.trailer_Lb[i-1] * math.tan(hitch_angle) * 
                math.sin(hitch_angle) / self.params.trailer_L[i-1]
            )
            curr_state['hitch_angle'] = hitch_angle
            
            # 更新位置
            hitch_x = prev_state['x'] - self.params.trailer_Lb[i-1] * math.cos(prev_state['yaw'])
            hitch_y = prev_state['y'] - self.params.trailer_Lb[i-1] * math.sin(prev_state['yaw'])
            
            curr_state['x_front'] = hitch_x
            curr_state['y_front'] = hitch_y
            
            # 【改进】加入游隙死区逻辑
            hitch_error = normalize_angle(prev_state['yaw'] - (curr_state['yaw'] - dyaw))
            deadband = _angle_deadband_for_clearance(_hitch_clearance_m(), self.params.trailer_L[i])
            if abs(hitch_error) < deadband:
                curr_state['yaw'] -= dyaw
            
            # 【改进】强制后轴沿自身heading方向运动
            curr_state['x'] += curr_state['v'] * math.cos(curr_state['yaw']) * self.dt
            curr_state['y'] += curr_state['v'] * math.sin(curr_state['yaw']) * self.dt
            
            # 【改进】位置校正
            dx = curr_state['x_front'] - curr_state['x']
            dy = curr_state['y_front'] - curr_state['y']
            dist = math.hypot(dx, dy)
            if dist > 0:
                correction = dist - self.params.trailer_L[i]
                curr_state['x'] += correction * (dx / dist)
                curr_state['y'] += correction * (dy / dist)
                curr_state['yaw'] = math.atan2(dy, dx)
        
        return new_tractor_state, new_trailer_states
