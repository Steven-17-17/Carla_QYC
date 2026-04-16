% 终端输入
% 关闭“仅预览 IK”模式（否则会提前 return）
% setenv('IK_LAST_ONLY_PREVIEW','0')
% 开启“最后一节挂车 IK 全程”模式
% setenv('RUN_LAST_TRAILER_IK_FULLRUN','1')
% 切回原来的分阶段模式：
% setenv('RUN_LAST_TRAILER_IK_FULLRUN','0')

%function mpc_multi_trailer()
clc
clear
    % 建立基于NMPC（非线性模型预测控制）的多挂车直角转弯路径跟踪模型
    % 依赖：MATLAB 优化工具箱 (fmincon)
    
    dt = 0.1; 
    vpi = init_vehicle_param_info(8); % 设置为 8 节状态 (即代表 4 节完整挂车，1 dolly + 1 semi = 1 full trailer)
    % 预览开关：在 MATLAB 中执行 setenv('IK_LAST_ONLY_PREVIEW','1') 后再运行脚本
    preview_last_trailer_ik_only = strcmp(getenv('IK_LAST_ONLY_PREVIEW'), '1');
    
    % --- 1. 参考路径来源 ---
    % 旧版“手工构造直角转弯参考线”保留备查，当前不再启用
    %{
    R=10;  %转弯半径

    % 第一段：向东直行 (需要长一点，留足逆运动学初始迭代的空间)
    x1 = -30:0.1:R;
    y1 = zeros(size(x1));

    % 第二段：第一处右转圆弧 (半径10，转向南)
    arc_length1 = R * (pi/2);
    num_arc_steps1 = round(arc_length1 / 0.1);
    th1 = linspace(pi/2, 0, num_arc_steps1);
    th1 = th1(2:end);
    x2 = R + R*cos(th1);
    y2 = -R + R*sin(th1);

    % 第三段：向下直行 (x: 20, y: -10 -> -60)
    y3 = (y2(end)-0.1):-0.1:-30;
    x3 = 2*R * ones(size(y3));

    % 第四段：第二处右转圆弧 (由向南转为向西)
    % 起点是第三段终点 (2*R, -60)。圆心在它右侧即 (R, -60)
    arc_length2 = R * (pi/2);
    num_arc_steps2 = round(arc_length2 / 0.1);
    th2 = linspace(0, -pi/2, num_arc_steps2);
    th2 = th2(2:end);
    x4 = R + R*cos(th2);
    y4 = -30 + R*sin(th2);

    % 第五段：向西直行，保证车辆转弯后有路线能开出圆弧，并且彻底停稳 (这里延长30米)
    x5 = (x4(end)-0.1):-0.1:-30;
    y5 = y4(end) * ones(size(x5));

    % 整合整条路径
    last_trailer_ref_x = [x1, x2, x3, x4, x5];
    last_trailer_ref_y = [y1, y2, y3, y4, y5];
    %}

    % 新版：从 hybrid A* 参考路径文件加载 (CSV列: x, y, yaw)
    ref_path_file = 'hybrid_astar_path_0331_full.csv';
    if ~exist(ref_path_file, 'file')
        error('未找到参考路径文件: %s', ref_path_file);
    end
    ref_data = readmatrix(ref_path_file);
    if size(ref_data, 2) < 2
        error('参考路径文件格式错误：至少需要两列 x,y。');
    end
    last_trailer_ref_x = ref_data(:, 1)';
    last_trailer_ref_y = ref_data(:, 2)';

    % 清除 NaN 与重复点，避免后续曲率/误差计算出现数值问题
    valid_mask = ~isnan(last_trailer_ref_x) & ~isnan(last_trailer_ref_y);
    last_trailer_ref_x = last_trailer_ref_x(valid_mask);
    last_trailer_ref_y = last_trailer_ref_y(valid_mask);
    if numel(last_trailer_ref_x) >= 2
        step_len = hypot(diff(last_trailer_ref_x), diff(last_trailer_ref_y));
        keep_mask = [true, step_len > 1e-6];
        last_trailer_ref_x = last_trailer_ref_x(keep_mask);
        last_trailer_ref_y = last_trailer_ref_y(keep_mask);
    end
    if numel(last_trailer_ref_x) < 2
        error('加载后的参考路径点数不足，请检查文件: %s', ref_path_file);
    end
    disp(['Loaded hybrid A* reference: ', ref_path_file, ', points = ', num2str(numel(last_trailer_ref_x))]);

    % 场景障碍物仍沿用当前地图参数
    R = 10;

    % --- 调试预览：只运行“最后一节挂车”逆运动学 ---
    if preview_last_trailer_ik_only
        [ik_last_x, ik_last_y, ~] = generate_inverse_kinematics_trajectory(...
            last_trailer_ref_x, last_trailer_ref_y, vpi, vpi.num_trailers);

        figure('Name', 'IK Preview - Last Trailer Only', 'Position', [120, 120, 1100, 800]);
        ax_dbg = gca;
        hold(ax_dbg, 'on');
        axis(ax_dbg, 'equal');
        grid(ax_dbg, 'on');

        plot(ax_dbg, last_trailer_ref_x, last_trailer_ref_y, 'k--', 'LineWidth', 2.8, ...
            'DisplayName', 'Input Ref (Last Trailer)');
        plot(ax_dbg, ik_last_x, ik_last_y, 'r-', 'LineWidth', 2.0, ...
            'DisplayName', 'IK Output (Tractor)');

        title(ax_dbg, sprintf('Last Trailer IK Preview | points in: %d | points out: %d', ...
            numel(last_trailer_ref_x), numel(ik_last_x)));
        xlabel(ax_dbg, 'X (m)');
        ylabel(ax_dbg, 'Y (m)');
        legend(ax_dbg, 'show', 'Location', 'best');

        disp('Preview mode: 已完成“最后一节挂车”IK轨迹绘制，脚本提前结束。');
        return;
    end
    
    % --- 逆运动学反演：从指定挂车目标反推牵引车目标轨迹 ---
    % 全程模式开关：默认关闭（即分阶段模式）。可用环境变量 RUN_LAST_TRAILER_IK_FULLRUN 覆盖（'1'开/'0'关）
    fullrun_env = getenv('RUN_LAST_TRAILER_IK_FULLRUN');
    if isempty(fullrun_env)
        run_last_trailer_ik_fullrun = false;
    else
        run_last_trailer_ik_fullrun = strcmp(fullrun_env, '1');
    end
    if run_last_trailer_ik_fullrun
        control_target_sequence = vpi.num_trailers;
    else
        % 分阶段切换受控挂车：2 -> 4 -> 6 -> 8
        control_target_sequence = [2, 4, 6, 8];
    end
    target_stage_idx = 1;
    target_controlled_trailer_idx = control_target_sequence(target_stage_idx);
    
    % 给定挂车的理想路径，算出能让它正好压线的牵引车特殊路径
    [ref_x, ref_y, ref_yaw] = generate_inverse_kinematics_trajectory(...
        last_trailer_ref_x, last_trailer_ref_y, vpi, target_controlled_trailer_idx);
    
    % 起点为了美观，我们把时间截断到真正需要控制的地方（牵引车 x >= 0）
    start_idx = find(ref_x >= 0, 1, 'first');
    if isempty(start_idx), start_idx = 1; end
    % 将反推的参考轨迹裁剪对齐
    ref_x = ref_x(start_idx:end);
    ref_y = ref_y(start_idx:end);
    ref_yaw = ref_yaw(start_idx:end);

    % 预生成每个阶段(2/4/6/8节)的参考轨迹，便于可视化全阶段路径
    num_stages = length(control_target_sequence);
    stage_ref_x = cell(1, num_stages);
    stage_ref_y = cell(1, num_stages);
    stage_ref_yaw = cell(1, num_stages);
    stage_ref_x{1} = ref_x;
    stage_ref_y{1} = ref_y;
    stage_ref_yaw{1} = ref_yaw;
    for s = 2:num_stages
        [sx, sy, syaw] = generate_inverse_kinematics_trajectory(...
            last_trailer_ref_x, last_trailer_ref_y, vpi, control_target_sequence(s));
        s_start_idx = find(sx >= 0, 1, 'first');
        if isempty(s_start_idx), s_start_idx = 1; end
        stage_ref_x{s} = sx(s_start_idx:end);
        stage_ref_y{s} = sy(s_start_idx:end);
        stage_ref_yaw{s} = syaw(s_start_idx:end);
    end
    % 预生成“牵引车头引导”参考轨迹：直接以黑色目标线作为车头参考
    tractor_head_ref_x = last_trailer_ref_x;
    tractor_head_ref_y = last_trailer_ref_y;
    tractor_head_ref_yaw = calc_path_yaw_from_xy(tractor_head_ref_x, tractor_head_ref_y);
    tractor_start_idx = find(tractor_head_ref_x >= 0, 1, 'first');
    if isempty(tractor_start_idx), tractor_start_idx = 1; end
    tractor_head_ref_x = tractor_head_ref_x(tractor_start_idx:end);
    tractor_head_ref_y = tractor_head_ref_y(tractor_start_idx:end);
    tractor_head_ref_yaw = tractor_head_ref_yaw(tractor_start_idx:end);
    is_tractor_head_guided = false;
    
    % --- 2. 初始车辆状态 ---
    % 【新增】设置每节挂车的初始偏航角度（相对于前一节车的角度差，单位：弧度）
    % 正值代表在上一节车方向的左侧，负值代表右侧，0代表完全对齐
    % 例如：init_rel_angles(1) = 0.1 代表第一节挂车相对牵引车向左偏 0.1 rad
    init_rel_angles = zeros(1, vpi.num_trailers); 
    % init_rel_angles(1) = -0.3;
    % init_rel_angles(2) = -0.3;
    % init_rel_angles(3) = -0.3;
    % init_rel_angles(4) = -0.3;
    % init_rel_angles(5) = -1.2;
    % init_rel_angles(7) = -1.2;
    
    % 放置在逆运动学推出的起点上
    vehicle = init_vehicle(ref_x(1), ref_y(1), ref_yaw(1), 1.5, dt, vpi, init_rel_angles); 
    
    % --- 3. MPC 参数 ---
    Np_base = 15; % 基准预测时域
    Np_min = 8;   % 弯道采用更短时域，降低过度前瞻带来的摆振
    Np_max = 22;  % 直线采用更长时域，增强前瞻稳定性
    curv_soft = 0.015; % 曲率软阈值 (1/m)
    curv_hard = 0.09;  % 曲率硬阈值 (1/m)
    w_y = 10.0; % 稍微降一点对坐标的死板要求
    w_yaw = 1.0; % 车头航向追踪，让车头更自由，避免因微小角度波动而引起转向系统共振
    w_delta = 0.5; % 增加打死方向盘的成本
    w_ddelta = 50.0; % 重罚打方向盘的速度，防止“画龙”
    w_trailer = 0.0; % 关闭预测惩罚，牵引车只要走好 IK 轨迹，尾车必定完美入库！
    
    % 初始猜测（会在主循环中按动态 Np 自动缩放）
    u0 = zeros(Np_base, 1);
    options = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp');
    
    % --- 4. 仿真与绘图准备 ---
    figure('Position', [100, 100, 1200, 933]);
    ax = gca; hold(ax, 'on'); axis(ax, 'equal');
    stage_path_colors = lines(num_stages);
    
    % 画出尾车的最终真理约束线（黑色加粗虚线）
    plot(ax, last_trailer_ref_x, last_trailer_ref_y, 'k--', 'LineWidth', 3, 'DisplayName', 'Target Path (Last Trailer)');
    % 画出所有阶段的IK参考轨迹（细虚线）
    for s = 1:num_stages
        plot(ax, stage_ref_x{s}, stage_ref_y{s}, '--', 'Color', stage_path_colors(s, :), ...
            'LineWidth', 1.0, 'DisplayName', sprintf('IK Ref (Trailer %d)', control_target_sequence(s)));
    end
    % 画出 NMPC 追踪的牵引车特制反推路线（红色细实线）
    plot(ax, ref_x, ref_y, 'r-', 'LineWidth', 1.6, 'DisplayName', 'Active IK Path (Tractor)');
    
    % 其他障碍物暂时注释（墙体与矩形）
    %{
    % 添加两侧的直线墙壁障碍物
    wall_left_x = [2*R - 1.8, 2*R - 1.8];
    wall_left_y = [-15, -30];
    wall_right_x = [2*R + 1.8, 2*R + 1.8];
    wall_right_y = [-15, -30];
    plot(ax, wall_left_x, wall_left_y, 'k-', 'LineWidth', 2, 'Color', [0.3 0.3 0.3], 'DisplayName', 'Left Wall');
    plot(ax, wall_right_x, wall_right_y, 'k-', 'LineWidth', 2, 'Color', [0.3 0.3 0.3], 'DisplayName', 'Right Wall');
    
    % 在右侧增设一个向右扩展的矩形障碍物
    % 矩形左上角为 (2*R + 1.8, -25)，宽 4m (Y方向延伸)，长 2m (X方向延伸向右)
    rect_x_min = 2*R + 1.8;
    rect_x_max = 2*R + 1.8 + 2.0;
    rect_y_max = -23;
    rect_y_min = -23 - 4.0;
    
    % 绘制矩形障碍物
    rectangle('Position', [rect_x_min, rect_y_min, 2.0, 4.0], 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', 'k', 'LineWidth', 1.5, 'Parent', ax);
    %}
    
    color_list = {'r', 'b', 'g', 'm', 'c', 'k', 'y'};
    
    % --- 视频录制准备 ---
    video_filename = 'mpc_tracking_animation.mp4';
    vidObj = VideoWriter(video_filename, 'MPEG-4');
    vidObj.FrameRate = 15; % 设置视频帧率
    open(vidObj);
    
    history_x = []; history_y = [];
    history_last_trailer_x = []; history_last_trailer_y = []; % 【新增】记录最后一节挂车的历史轨迹
    history_v = []; history_error = []; history_tractor_to_black_err = []; time_history = [];
    history_solve_time = []; % 记录每次MPC求解时间
    history_trailer_errors = cell(1, vpi.num_trailers);
    switch_event_times = []; % 记录阶段切换时刻（用于误差图竖线）
    switch_event_trailer_idx = []; % 记录在该时刻完成对齐的挂车节号
    sim_time = 0;
    
    % --- 定义平滑的目标速度曲线生成函数 ---
    % 构建一个起步加速、匀速、最后可能减速的平滑曲线
    % 此处使用一个简单的正弦平滑过渡：从 0 到 max_v
    target_v_max = 1.5; % 最高目标速度 1.5m/s
    accel_time = 5.0;   % 加速所需时间 5s
    get_target_v = @(t) target_v_max * (0.5 - 0.5 * cos(pi * min(t/accel_time, 1)));
    
    % --- 纵向控制参数初始设置 ---
    % 待填写的物理质量（用户提供）
    m_tractor = 8000;       % 牵引车质量 (kg) - 请替换为实际值
    m_trailer_empty = 5000; % 单节挂车空载质量 (kg) - 请替换为实际值
    num_full_trailers = vpi.num_trailers / 2; % 满挂车数量 (1 Dolly + 1 Semi = 1 挂)
    total_payload = 0;      % 额外货物载重 (kg)
    m_total = m_tractor + num_full_trailers * m_trailer_empty + total_payload;
    
    % 自适应PID状态变量
    I_ev = 0.0;      % 积分累加器
    prev_ev = 0.0;   % 上一时刻速度误差
    
    % 障碍物避让停滞状态机变量
    obs_stop_state = 0; % 0: 正常行驶, 1: 准备停车, 2: 停滞等待中, 3: 恢复行驶
    obs_stop_timer = 0;
    obs_resume_time = 0;
    obs_x_target = 10.5;
    obs_y_target = 18.0;
    obs_stop_radius = 0.2;
    obs_stop_radius_soft = 0.45;   % 软阈值：允许接近后低速触发停车
    obs_force_stop_radius = 0.9;   % 兜底半径：长时间无法命中硬阈值时允许强制进入等待
    obs_force_stop_speed = 0.12;   % 兜底速度阈值
    obs_stage1_timeout = 8.0;      % 停靠阶段最大允许持续时间 (s)
    obs_stage1_enter_time = -inf;
    final_stage_trigger_back_pts = 20; % 最后阶段：距离参考末端若干点内强制进入停车流程
    final_stage_stop_radius = 0.6;     % 最后阶段：以当前参考终点为准的停车半径
    obs_decel_dist = 5.0;
    obs_wait_time = 3.0;
    decel_dist_end = 5.0; % 距离终点5米开始减速
    switch_align_back_pts = 40;  % 阶段切换对齐窗口（向后）
    switch_align_fwd_pts = 80;   % 阶段切换对齐窗口（向前）
    
    % 切换瞬间转向平滑参数（不改参考线，只约束控制量连续）
    switch_steer_hold_steps = 15;   % 切换后保护窗口步数
    switch_steer_hold_counter = 0;
    max_steer_rate_normal = 1.2;    % 正常最大转角变化率 (rad/s)
    max_steer_rate_switch = 0.28;   % 切换窗口最大转角变化率 (rad/s)
    w_ddelta_boost_factor = 2.0;    % 切换窗口内增大转角变化惩罚

    % 参考点搜索与前视参数（抑制直线段来回切线导致的画龙）
    prev_min_idx = 1;
    search_back_pts = 3;
    search_forward_pts = 120;
    lookahead_pts_base = 6;
    curv_window_half = 12;
    
    % --- 5. 主循环 ---
    for step = 1:1200
        % 找到当前车辆在参考轨迹上的最近点索引（限窗+单调前进，避免索引抖动）
        if prev_min_idx > length(ref_x)
            prev_min_idx = 1;
        end
        search_start = max(1, prev_min_idx - search_back_pts);
        search_end = min(length(ref_x), prev_min_idx + search_forward_pts);
        [~, local_idx] = min((ref_x(search_start:search_end) - vehicle.x).^2 + ...
                             (ref_y(search_start:search_end) - vehicle.y).^2);
        min_idx = search_start + local_idx - 1;
        min_idx = max(min_idx, prev_min_idx);
        prev_min_idx = min_idx;
        
        % 按当前引导模式，计算“被引导对象”距当前参考终点的误差
        if is_tractor_head_guided
            guide_x = vehicle.x;
            guide_y = vehicle.y;
        else
            guide_x = vehicle.trailer_states(target_controlled_trailer_idx).x;
            guide_y = vehicle.trailer_states(target_controlled_trailer_idx).y;
        end
        ref_end_x = ref_x(end);
        ref_end_y = ref_y(end);
        guide_dist_to_end = sqrt((guide_x - ref_end_x)^2 + (guide_y - ref_end_y)^2);

        is_last_trailer_stage = (~run_last_trailer_ik_fullrun) && (~is_tractor_head_guided) && ...
                                (target_stage_idx >= length(control_target_sequence));
        
        % 终止条件改为：牵引车走到尽头，或者受控挂车已经非常接近最终点
        if min_idx > length(ref_x) - 1 || guide_dist_to_end < 0.2
            % 分段模式且尚未切换到车头引导时，不要提前退出，先让停靠状态机完成最终切换
            if run_last_trailer_ik_fullrun || is_tractor_head_guided
                break;
            end
        end
        
        % ---------------- 自适应PID纵向控制  ----------------
        % 1. 点位停靠停滞逻辑
        % 引导模式不同，点位距离判据的对象也不同
        if is_tractor_head_guided
            guide_x_for_obs = vehicle.x;
            guide_y_for_obs = vehicle.y;
        else
            controlled_trailer = vehicle.trailer_states(target_controlled_trailer_idx);
            guide_x_for_obs = controlled_trailer.x;
            guide_y_for_obs = controlled_trailer.y;
        end
        dist_to_obs = hypot(guide_x_for_obs - obs_x_target, guide_y_for_obs - obs_y_target);
        
        near_ref_end_by_idx = min_idx >= max(2, length(ref_x) - final_stage_trigger_back_pts);
        if obs_stop_state == 0 && (dist_to_obs < obs_decel_dist || ...
                                   (is_last_trailer_stage && (guide_dist_to_end < decel_dist_end || near_ref_end_by_idx)))
            obs_stop_state = 1; % 进入减速停车状态
            obs_stage1_enter_time = sim_time;
        end
        
        if obs_stop_state == 1
            stage1_elapsed = sim_time - obs_stage1_enter_time;
            if is_last_trailer_stage
                stop_dist_metric = min(dist_to_obs, guide_dist_to_end);
                stop_radius_hard = final_stage_stop_radius;
                stop_radius_soft = max(obs_stop_radius_soft, final_stage_stop_radius + 0.2);
            else
                stop_dist_metric = dist_to_obs;
                stop_radius_hard = obs_stop_radius;
                stop_radius_soft = obs_stop_radius_soft;
            end

            % 停靠点前减速
            if stop_dist_metric < stop_radius_hard
                target_v = 0.0;
            elseif stop_dist_metric < stop_radius_soft && abs(vehicle.v) < obs_force_stop_speed
                % 软阈值：已非常接近且速度足够低，也允许进入停车
                target_v = 0.0;
            else
                target_v = target_v_max * (stop_dist_metric / obs_decel_dist);
                target_v = max(target_v, 0.15); % 维持爬行直到抵达点位
            end
            
            % 如果要求停车且车基本停稳，进入停滞状态
            should_enter_wait = (target_v <= 1e-6 && abs(vehicle.v) < 0.05);
            if ~should_enter_wait
                fallback_near_stop = (dist_to_obs < obs_force_stop_radius) && (abs(vehicle.v) < obs_force_stop_speed);
                fallback_last_stage = is_last_trailer_stage && ((guide_dist_to_end < 0.8) || near_ref_end_by_idx);
                if stage1_elapsed >= obs_stage1_timeout && (fallback_near_stop || fallback_last_stage)
                    should_enter_wait = true;
                    disp(['停靠硬阈值未命中，触发软阈值/超时兜底进入等待态。当前时间: ', num2str(sim_time), 's']);
                end
            end

            if should_enter_wait
                obs_stop_state = 2;
                obs_stop_timer = sim_time;
                disp(['在停靠点(10.5,18)停车等待3秒... 当前时间: ', num2str(sim_time), 's']);
            end
            
        elseif obs_stop_state == 2
            target_v = 0.0;
            if (sim_time - obs_stop_timer) >= obs_wait_time
                if run_last_trailer_ik_fullrun
                    % 全程最后一节挂车IK模式：停靠后继续当前IK轨迹，不切阶段、不切车头引导
                    obs_stop_state = 3;
                    obs_resume_time = sim_time;
                    obs_stage1_enter_time = -inf;
                    disp(['最后一节挂车IK全程模式：停靠完成后继续当前IK轨迹。当前时间: ', num2str(sim_time), 's']);
                % 分阶段切换：2 -> 4 -> 6 -> 8，每次切换后重建牵引车参考轨迹
                elseif target_stage_idx < length(control_target_sequence)
                    aligned_trailer_idx = target_controlled_trailer_idx;
                    switch_event_times(end+1) = sim_time;
                    switch_event_trailer_idx(end+1) = aligned_trailer_idx;

                    target_stage_idx = target_stage_idx + 1;
                    target_controlled_trailer_idx = control_target_sequence(target_stage_idx);
                    new_ref_x = stage_ref_x{target_stage_idx};
                    new_ref_y = stage_ref_y{target_stage_idx};
                    new_ref_yaw = stage_ref_yaw{target_stage_idx};
                    
                    % 用“当前阶段进度索引附近”的局部搜索对齐，避免跨段跳到错误支路
                    align_search_lo = max(1, min_idx - switch_align_back_pts);
                    align_search_hi = min(length(new_ref_x), min_idx + switch_align_fwd_pts);
                    if align_search_hi > align_search_lo
                        [~, align_local] = min((new_ref_x(align_search_lo:align_search_hi) - vehicle.x).^2 + ...
                                               (new_ref_y(align_search_lo:align_search_hi) - vehicle.y).^2);
                        align_idx = align_search_lo + align_local - 1;
                    else
                        [~, align_idx] = min((new_ref_x - vehicle.x).^2 + (new_ref_y - vehicle.y).^2);
                    end
                    align_idx = max(1, min(align_idx, length(new_ref_x)));
                    ref_x = new_ref_x(align_idx:end);
                    ref_y = new_ref_y(align_idx:end);
                    ref_yaw = new_ref_yaw(align_idx:end);
                    if numel(ref_x) < 2
                        ref_x = new_ref_x;
                        ref_y = new_ref_y;
                        ref_yaw = new_ref_yaw;
                    end
                    
                    % 切换参考后保留当前转角连续性，避免前轮指令瞬时翻转
                    u0 = vehicle.steer * ones(size(u0));
                    switch_steer_hold_counter = switch_steer_hold_steps;
                    prev_min_idx = 1;
                    obs_stop_state = 0;
                    obs_stop_timer = 0;
                    obs_stage1_enter_time = -inf;
                    disp(['切换受控挂车到第', num2str(target_controlled_trailer_idx), '节，已重建IK参考轨迹。当前时间: ', num2str(sim_time), 's']);
                else
                    % 最后一节挂车完成后，切回牵引车头引导控制
                    is_tractor_head_guided = true;
                    new_ref_x = tractor_head_ref_x;
                    new_ref_y = tractor_head_ref_y;
                    new_ref_yaw = tractor_head_ref_yaw;
                    align_search_lo = max(1, min_idx - switch_align_back_pts);
                    align_search_hi = min(length(new_ref_x), min_idx + switch_align_fwd_pts);
                    if align_search_hi > align_search_lo
                        [~, align_local] = min((new_ref_x(align_search_lo:align_search_hi) - vehicle.x).^2 + ...
                                               (new_ref_y(align_search_lo:align_search_hi) - vehicle.y).^2);
                        align_idx = align_search_lo + align_local - 1;
                    else
                        [~, align_idx] = min((new_ref_x - vehicle.x).^2 + (new_ref_y - vehicle.y).^2);
                    end
                    align_idx = max(1, min(align_idx, length(new_ref_x)));
                    ref_x = new_ref_x(align_idx:end);
                    ref_y = new_ref_y(align_idx:end);
                    ref_yaw = new_ref_yaw(align_idx:end);
                    if numel(ref_x) < 2
                        ref_x = new_ref_x;
                        ref_y = new_ref_y;
                        ref_yaw = new_ref_yaw;
                    end
                    u0 = vehicle.steer * ones(size(u0));
                    switch_steer_hold_counter = switch_steer_hold_steps;
                    prev_min_idx = 1;
                    obs_stop_state = 3;
                    obs_resume_time = sim_time;
                    obs_stage1_enter_time = -inf;
                    disp(['最后一节挂车已停稳，切回牵引车头引导控制。当前时间: ', num2str(sim_time), 's']);
                end
            end
            
        elseif obs_stop_state == 3
            % 恢复行驶，重新平滑加速
            target_v = target_v_max * (0.5 - 0.5 * cos(pi * min((sim_time - obs_resume_time)/accel_time, 1)));
            
        else
            % 正常行驶 (包含刚起步)
            target_v = get_target_v(sim_time);
        end
        
        % 2. 终点停车逻辑 (优先级最高，覆盖上述所有规划)
        if guide_dist_to_end < decel_dist_end
            target_v = target_v_max * (guide_dist_to_end / decel_dist_end);
            % 最后阶段限制最低爬行速度，以防停机点前驻留停死
            if guide_dist_to_end < 3
                target_v = 0; % 最后几十厘米直接给0驻车
            else
                target_v = max(target_v, 0.2); 
            end
        end

        % 3) 速度上限保护：按铰接角自适应降速 + 全局硬上限，抑制后续挂车对不齐和末段飞车
        hitch_angles = arrayfun(@(ts) abs(normalize_angle(ts.hitch_angle)), vehicle.trailer_states);
        max_hitch = max(hitch_angles);
        hitch_soft = deg2rad(12);
        hitch_hard = deg2rad(28);
        if max_hitch <= hitch_soft
            hitch_factor = 1.0;
        elseif max_hitch >= hitch_hard
            hitch_factor = 0.35;
        else
            ratio = (max_hitch - hitch_soft) / max(hitch_hard - hitch_soft, 1e-6);
            hitch_factor = 1.0 - 0.65 * ratio;
        end
        target_v_cap = min(target_v_max, vpi.V_MAX) * hitch_factor;
        target_v_cap = max(target_v_cap, 0.15);
        target_v = min(target_v, target_v_cap);
        
        e_v = target_v - vehicle.v;
        delta_ev = e_v - prev_ev;
        prev_ev = e_v;
        
        % (1) 基础控制项 (空载电门开度需求拟合)
        % 公式: u_base = a*v_x^2 + b*v_x + c
        param_a = 0.00; % 需实验拟合
        param_b = 0.00; % 需实验拟合
        param_c = 0.00; % 需实验拟合
        u_base = param_a * vehicle.v^2 + param_b * vehicle.v + param_c;
        
        % (2) 挂车重量补偿项
        % 公式: u_w = a * (m_total - b)
        w_a = 0.0000; % 需实验标定 (电门开度增加量/kg)
        w_b = 20000;  % 免补偿阈值质量 (kg)    
        u_w = w_a * (m_total - w_b);
        
        % (3) 比例微分项
        % 公式: u_pd = kp * e_v + kd * delta_ev
        kp_v = 1.0; 
        kd_v = 0.1;
        u_PD = kp_v * e_v + kd_v * delta_ev;
        
        % (4) 带遗忘因子积分项 (消除累积误差)
        % 公式: u_I = 0.015 * (0.95 * I_{k-1} + e_{v,k})
        I_ev = 0.015 * (0.95 * I_ev + e_v);
        I_ev = min(max(I_ev, -0.8), 0.8); % 抗积分饱和，避免末段速度抬升
        u_I = I_ev;
        
        % 总电门开度 (此处以加速度替代)
        u_total = u_base + u_w + u_PD + u_I;
        
        % 对应到油门/刹车开度指令 (-1 到 1)，在仿真中暂时将其映射为加速度指令
        accel_cmd = min(max(u_total, -1.0), 1.0); % 限幅操作，避免异常加速
        % -------------------------------------------------------------------
        
        % 基于局部曲率自适应选择预测时域：弯道缩短，直线拉长
        local_kappa = estimate_local_curvature(ref_x, ref_y, ref_yaw, min_idx, curv_window_half);
        if local_kappa <= curv_soft
            Np = Np_max;
        elseif local_kappa >= curv_hard
            Np = Np_min;
        else
            kappa_ratio = (local_kappa - curv_soft) / max(curv_hard - curv_soft, 1e-6);
            Np = round(Np_max - kappa_ratio * (Np_max - Np_min));
        end
        Np = max(Np_min, min(Np_max, Np));
        u0 = resize_warm_start(u0, Np, vehicle.steer);
        lb = -vpi.MAX_STEER * ones(Np, 1);
        ub =  vpi.MAX_STEER * ones(Np, 1);

        % 截取未来 Np 个点的参考轨迹（加入前视，减小对最近点抖动的敏感性）
        lookahead_pts = lookahead_pts_base + round(max(vehicle.v, 0) * 2);
        lookahead_pts = min(max(lookahead_pts, 4), 10);
        target_start_idx = min(min_idx + lookahead_pts, length(ref_x));
        end_idx = target_start_idx + Np - 1;
        if end_idx <= length(ref_x)
            target_x = ref_x(target_start_idx : end_idx);
            target_y = ref_y(target_start_idx : end_idx);
            target_yaw = ref_yaw(target_start_idx : end_idx);
        else
            % 如果接近终点，预测步数超出了参考轨迹长度，则进行终点补齐（沿切向延长，而不是原地重复！）
            pad_len = end_idx - length(ref_x);
            
            % 取最后两个点计算延伸方向
            dx_end = ref_x(end) - ref_x(end-1);
            dy_end = ref_y(end) - ref_y(end-1);
            
            extend_x = ref_x(end) + dx_end * (1:pad_len);
            extend_y = ref_y(end) + dy_end * (1:pad_len);
            
            target_x = [ref_x(target_start_idx:end), extend_x];
            target_y = [ref_y(target_start_idx:end), extend_y];
            target_yaw = [ref_yaw(target_start_idx:end), repmat(ref_yaw(end), 1, pad_len)];
        end
        
        w_y_eff = w_y;
        w_yaw_eff = w_yaw;
        w_ddelta_eff = w_ddelta;
        if is_tractor_head_guided
            % 牵引车头引导阶段提高航向阻尼，降低来回穿线振荡
            w_yaw_eff = 2.0 * w_yaw;
        end
        if switch_steer_hold_counter > 0
            w_ddelta_eff = w_ddelta * w_ddelta_boost_factor;
        end

        % 定义目标函数 (移除了对 local_ref_x/y 的依赖)
        cost_func = @(U) mpc_cost(U, vehicle, target_x, target_y, target_yaw, ...
                                  w_y_eff, w_yaw_eff, w_delta, w_ddelta_eff, Np, dt);
        
        % 记录求解开始时间
        solve_tic = tic;
        
        % 求解最优控制序列 U (前轮转角)
        [U_opt, ~] = fmincon(cost_func, u0, [], [], [], [], lb, ub, [], options);
        
        % 记录求解结束时间并保存
        solve_time = toc(solve_tic);
        history_solve_time(end+1) = solve_time;
        
        % 对整条控制序列施加与执行端一致的转角变化率约束，避免模型-执行不一致诱发极限环
        if switch_steer_hold_counter > 0
            steer_rate_limit = max_steer_rate_switch;
        else
            steer_rate_limit = max_steer_rate_normal;
        end
        max_steer_step = steer_rate_limit * dt;
        U_cmd = apply_steer_rate_limit(U_opt, vehicle.steer, max_steer_step, vpi.MAX_STEER);
        
        % 将限幅后的控制序列移位供下一步作为初始值
        u0 = [U_cmd(2:end); U_cmd(end)];
        
        % 下发第一个控制量
        delta_cmd = U_cmd(1);
        if switch_steer_hold_counter > 0
            switch_steer_hold_counter = switch_steer_hold_counter - 1;
        end
        
        % 更新真实车辆状态 (下发加速度指令，进行纵向变速)
        vehicle = update_vehicle(vehicle, accel_cmd, delta_cmd);
        
        % 计算当前真实车辆(牵引车)与参考轨迹的最短距离(精确的横向跟踪误差，依据红线)
        tracking_err = calc_cross_track_error(vehicle.x, vehicle.y, ref_x, ref_y);
        
        % 计算当前真实车辆(牵引车)与【原本黑虚线】的距离误差
        tractor_to_black_err = calc_cross_track_error(vehicle.x, vehicle.y, last_trailer_ref_x, last_trailer_ref_y);
        
        sim_time = sim_time + dt;
        
        % 记录牵引车与测试时间
        history_x(end+1) = vehicle.x;
        history_y(end+1) = vehicle.y;
        history_v(end+1) = vehicle.v;
        history_error(end+1) = tracking_err;
        history_tractor_to_black_err(end+1) = tractor_to_black_err;
        time_history(end+1) = sim_time;
        
        % 【更新】记录受控挂车的历史轨迹
        history_last_trailer_x(end+1) = vehicle.trailer_states(target_controlled_trailer_idx).x;
        history_last_trailer_y(end+1) = vehicle.trailer_states(target_controlled_trailer_idx).y;
        
        % 记录所有挂车的后轴中心与【原本的尾车参考线】的跟踪误差
        for t_idx = 1:vpi.num_trailers
            trailer_err = calc_cross_track_error(vehicle.trailer_states(t_idx).x, vehicle.trailer_states(t_idx).y, last_trailer_ref_x, last_trailer_ref_y);
            history_trailer_errors{t_idx}(end+1) = trailer_err;
        end
        
        if mod(step, 5) == 0
            cla(ax);
            % 画出尾车的最终真理约束线（黑色加粗虚线）
            plot(ax, last_trailer_ref_x, last_trailer_ref_y, 'k--', 'LineWidth', 3);
            % 画出所有阶段的IK参考轨迹（细虚线）
            for s = 1:num_stages
                plot(ax, stage_ref_x{s}, stage_ref_y{s}, '--', 'Color', stage_path_colors(s, :), 'LineWidth', 1.0);
            end
            % 画出 NMPC 追踪的牵引车特制反推路线（红色细实线）
            plot(ax, ref_x, ref_y, 'r-', 'LineWidth', 1.6);
            
            % 其他障碍物暂时注释（墙体与矩形）
            %{
            % 画出场景地图中的墙壁障碍物
            plot(ax, wall_left_x, wall_left_y, 'k-', 'LineWidth', 2, 'Color', [0.3 0.3 0.3]);
            plot(ax, wall_right_x, wall_right_y, 'k-', 'LineWidth', 2, 'Color', [0.3 0.3 0.3]);
            
            % 在每一帧中也要重绘这个矩形障碍物
            rectangle('Position', [rect_x_min, rect_y_min, 2.0, 4.0], 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', 'k', 'LineWidth', 1.5, 'Parent', ax);
            %}
            
            % 画出车头实际走出的轨迹
            plot(ax, history_x, history_y, 'g-', 'LineWidth', 1.5);
            
            % 【新增】画出最后一节挂车实际走出的轨迹（蓝色实线，以此来直观对比它和黑色虚线的贴合度）
            plot(ax, history_last_trailer_x, history_last_trailer_y, 'b-', 'LineWidth', 2);
            
            draw_vehicle_n_trailers(vehicle, ax, color_list);
            if is_tractor_head_guided
                guide_label = 'Guide: Tractor Head';
            else
                guide_label = sprintf('Guide: Trailer %d', target_controlled_trailer_idx);
            end
            title(sprintf('IK-NMPC Stage %d/%d | %s | Step: %d', ...
                target_stage_idx, length(control_target_sequence), guide_label, step));
            
            % 动态计算包围框以包含所有拖车
            min_x = vehicle.x - 15;
            max_x = vehicle.x + 15;
            min_y = vehicle.y - 15;
            max_y = vehicle.y + 15;
            for t_idx = 1:vpi.num_trailers
                min_x = min(min_x, vehicle.trailer_states(t_idx).x - 15);
                max_x = max(max_x, vehicle.trailer_states(t_idx).x + 15);
                min_y = min(min_y, vehicle.trailer_states(t_idx).y - 15);
                max_y = max(max_y, vehicle.trailer_states(t_idx).y + 15);
            end
            axis([min_x, max_x, min_y, max_y]);
            
            drawnow limitrate;
            
            % 将当前画面抓取并写入到视频帧中
            frame = getframe(gcf);
            writeVideo(vidObj, frame);
        end
    end
    disp('MPC Tracking finished!');
    close(vidObj); % 循环结束，关闭录像文件
    disp(['Video saved to ', video_filename]);
    
    % --- 在控制台打印最终引导对象的停车误差 ---
    ref_end_x = ref_x(end);
    ref_end_y = ref_y(end);
    
    fprintf('\n================== 最终停车误差分析 ==================\n');
    if is_tractor_head_guided
        final_distance_to_target = sqrt((vehicle.x - ref_end_x)^2 + (vehicle.y - ref_end_y)^2);
        final_cross_track_error = history_error(end);
        fprintf('牵引车头与当前引导轨迹终点的总体欧氏距离: %.4f m\n', final_distance_to_target);
        fprintf('牵引车头在最终时刻的横向跟踪误差:        %.4f m\n', final_cross_track_error);
    else
        last_t_x = vehicle.trailer_states(target_controlled_trailer_idx).x;
        last_t_y = vehicle.trailer_states(target_controlled_trailer_idx).y;
        final_distance_to_target = sqrt((last_t_x - ref_end_x)^2 + (last_t_y - ref_end_y)^2);
        final_cross_track_error = history_trailer_errors{target_controlled_trailer_idx}(end);
        fprintf('受控挂车(第%d节)后轴中心与理想终点的总体欧氏距离: %.4f m\n', target_controlled_trailer_idx, final_distance_to_target);
        fprintf('受控挂车(第%d节)在最终时刻真实的横向跟踪误差:      %.4f m\n', target_controlled_trailer_idx, final_cross_track_error);
    end
    fprintf('====================================================\n\n');
    
    % --- 输出速度和路径跟踪误差图 ---
    figure('Name', 'MPC Tracking Performance', 'Position', [150, 150, 800, 800]);
    
    % 速度子图
    subplot(2, 1, 1);
    plot(time_history, history_v, 'b', 'LineWidth', 2);
    grid on;
    xlabel('Time (s)');
    ylabel('Velocity (m/s)');
    title('Vehicle Velocity vs. Time');
    
    % 跟踪误差子图 (包含牵引车与挂车)
    subplot(2, 1, 2);
    % 牵引车与红线的误差 (控制指标)
    plot(time_history, history_error, 'r-', 'LineWidth', 2, 'DisplayName', 'Tractor vs IK Red Line');
    hold on;
    % 牵引车与黑线的误差 (对比指标，用红色虚线表示)
    plot(time_history, history_tractor_to_black_err, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Tractor vs Black Target Line');
    
    for t_idx = 2:2:vpi.num_trailers
        plot(time_history, history_trailer_errors{t_idx}, '--', 'DisplayName', sprintf('Trailer %d', t_idx/2));
    end
    % 在每次阶段切换时刻画竖直虚线，并标注“哪节挂车完成对齐”
    for ev_idx = 1:length(switch_event_times)
        xline(switch_event_times(ev_idx), 'k--', sprintf('Trailer %d aligned', switch_event_trailer_idx(ev_idx)/2), ...
            'LineWidth', 1.2, 'LabelVerticalAlignment', 'middle', 'LabelHorizontalAlignment', 'left');
    end
    % 画一条0误差基准线
    plot(time_history, zeros(size(time_history)), 'k:', 'LineWidth', 1.5, 'HandleVisibility','off');
    grid on;
    xlabel('Time (s)');
    ylabel('Signed Cross Track Error (m)');
    title('Path Tracking Error (+ Left / - Right) vs. Time');
    legend('show');
    
    % --- 导出误差数据到 CSV 文件 ---
    csv_filename = 'tracking_errors_data.csv';
    header = {'Time_s', 'Tractor_vs_RedLine_Error_m', 'Tractor_vs_BlackLine_Error_m'};
    for t_idx = 1:(vpi.num_trailers/2)
        header{end+1} = sprintf('Trailer_%d_Error_m', t_idx);
    end
    
    % 拼凑数据矩阵
    data_matrix = [time_history', history_error', history_tractor_to_black_err'];
    for t_idx = 2:2:vpi.num_trailers
        data_matrix = [data_matrix, history_trailer_errors{t_idx}'];
    end
    
    % 创建 table 并写入文件
    T = array2table(data_matrix, 'VariableNames', header);
    writetable(T, csv_filename);
    fprintf('Error data successfully saved to %s\n', csv_filename);
%end

function path_yaw = calc_path_yaw_from_xy(path_x, path_y)
    path_yaw = zeros(size(path_x));
    if numel(path_x) < 2
        return;
    end
    for k = 1:(numel(path_x)-1)
        path_yaw(k) = atan2(path_y(k+1)-path_y(k), path_x(k+1)-path_x(k));
    end
    path_yaw(end) = path_yaw(end-1);
    path_yaw = unwrap(path_yaw);
end

function cte = calc_cross_track_error(x, y, ref_x, ref_y)
    % 找到离输入点(x,y)最近的参考点
    [~, min_idx] = min((ref_x - x).^2 + (ref_y - y).^2);
    
    % 始终选择带有明确方向的线段 (idx1 -> idx2) 作为参考前进方向
    if min_idx == length(ref_x)
        idx1 = min_idx - 1;
        idx2 = min_idx;
    else
        idx1 = min_idx;
        idx2 = min_idx + 1;
    end
    
    x1 = ref_x(idx1); y1 = ref_y(idx1);
    x2 = ref_x(idx2); y2 = ref_y(idx2);
    
    dx = x2 - x1;
    dy = y2 - y1;
    
    % 计算点到线段的精确垂直距离（带符号）
    den = sqrt(dx^2 + dy^2);
    if den < 1e-6
        cte = sqrt((x - x1)^2 + (y - y1)^2);
    else
        % 利用叉乘计算有符号误差
        % 正数：车辆在参考轨迹的左侧
        % 负数：车辆在参考轨迹的右侧
        cte = (dx * (y - y1) - dy * (x - x1)) / den;
    end
end

function U_limited = apply_steer_rate_limit(U, steer_start, max_step, steer_abs_limit)
    U_limited = zeros(size(U));
    prev = max(-steer_abs_limit, min(steer_abs_limit, steer_start));
    for i = 1:numel(U)
        ui = max(-steer_abs_limit, min(steer_abs_limit, U(i)));
        ui = min(max(ui, prev - max_step), prev + max_step);
        U_limited(i) = ui;
        prev = ui;
    end
end

function u_out = resize_warm_start(u_in, new_len, fill_value)
    if nargin < 3
        fill_value = 0;
    end
    if isempty(u_in)
        u_out = fill_value * ones(new_len, 1);
        return;
    end
    u_in = u_in(:);
    if numel(u_in) == new_len
        u_out = u_in;
    elseif numel(u_in) > new_len
        u_out = u_in(1:new_len);
    else
        u_out = [u_in; repmat(u_in(end), new_len - numel(u_in), 1)];
    end
end

function kappa = estimate_local_curvature(path_x, path_y, path_yaw, center_idx, half_window)
    n = numel(path_x);
    if n < 3
        kappa = 0;
        return;
    end
    i1 = max(1, center_idx - half_window);
    i2 = min(n, center_idx + half_window);
    if i2 - i1 < 2
        kappa = 0;
        return;
    end

    seg_x = path_x(i1:i2);
    seg_y = path_y(i1:i2);
    seg_yaw = path_yaw(i1:i2);

    ds = hypot(diff(seg_x), diff(seg_y));
    ds = max(ds, 1e-4);

    dyaw = diff(seg_yaw);
    dyaw = arrayfun(@(a) abs(normalize_angle(a)), dyaw);

    kappa_samples = dyaw ./ ds;
    kappa_samples = kappa_samples(isfinite(kappa_samples));
    if isempty(kappa_samples)
        kappa = 0;
    else
        kappa = median(kappa_samples);
    end
end

% --- MPC 成本函数 ---
function cost = mpc_cost(U, current_vehicle, target_x, target_y, target_yaw, w_y, w_yaw, w_delta, w_ddelta, Np, dt)
    cost = 0;
    pred_vehicle = current_vehicle;
    last_delta = pred_vehicle.steer;
    vpi = pred_vehicle.vpi;
    
    for k = 1:Np
        % 使用预测的控制量只更新牵引车
        delta = max(-vpi.MAX_STEER, min(vpi.MAX_STEER, U(k)));
        pred_vehicle.steer = delta;
        % pred_vehicle.v = pred_vehicle.v; % 加速度为0
        pred_vehicle.yaw = normalize_angle(pred_vehicle.yaw + pred_vehicle.v / vpi.L * tan(delta) * dt);
        pred_vehicle.x = pred_vehicle.x + pred_vehicle.v * cos(pred_vehicle.yaw) * dt;
        pred_vehicle.y = pred_vehicle.y + pred_vehicle.v * sin(pred_vehicle.yaw) * dt;
        
        % 计算误差
        dx = pred_vehicle.x - target_x(k);
        dy = pred_vehicle.y - target_y(k);
        
        % 横向误差 (简化计算，用距离平方)
        dist_err2 = dx^2 + dy^2;
        
        % 航向角误差
        yaw_err = normalize_angle(pred_vehicle.yaw - target_yaw(k));
        
        % 控制变化率
        ddelta = U(k) - last_delta;
        
        % 累加代价
        cost = cost + w_y * dist_err2 + w_yaw * yaw_err^2 + ...
               w_delta * U(k)^2 + w_ddelta * ddelta^2;
               
        last_delta = U(k);
    end
end


% 因空间关系，以下直接嵌入基础车辆物理计算部分

function vpi = init_vehicle_param_info(num_trailers)
    vpi.L = 2.0; vpi.W = 2.0; vpi.LF = 2.7; vpi.LB = 0.8;
    vpi.MAX_STEER = 0.6; vpi.TR = 0.5; vpi.TW = 0.5; 
    vpi.V_MAX = 1.8;    % 车辆速度硬上限 (m/s)
    vpi.V_MIN = -0.3;   % 车辆最小速度 (m/s)，允许轻微倒滑
    % 为了保证轮胎最外侧不超过总车宽vpi.W，轮毂中心距 WD 应等于 W - 轮胎宽度 (TW)
    vpi.WD = vpi.W - vpi.TW; 
    vpi.LENGTH = vpi.LB + vpi.LF; vpi.La = 2.7; vpi.Lb = 0.8;
    vpi.num_trailers = num_trailers;
    vpi.trailer_L = zeros(1, num_trailers);
    vpi.trailer_Lb = zeros(1, num_trailers);
    for i = 1:num_trailers
        if mod(i, 2) == 1
            vpi.trailer_L(i) = 1.4; vpi.trailer_Lb(i) = 0;
        else
            vpi.trailer_L(i) = 2.0; vpi.trailer_Lb(i) = 0.5;
        end
    end
end

function a = normalize_angle(angle)
    a = mod(angle + pi, 2 * pi);
    if a < 0.0, a = a + 2 * pi; end
    a = a - pi;
end

function c = hitch_clearance_m()
    c = max(0.0, 0.04 - 0.02);
end

function db = angle_deadband_for_clearance(clearance_m, lever_m)
    db = atan2(clearance_m, max(lever_m, 1e-6));
end

function vehicle = init_vehicle(x, y, yaw, v, dt, vpi, init_rel_angles)
    if nargin < 7
        init_rel_angles = zeros(1, vpi.num_trailers);
    end

    vehicle.steer = 0; vehicle.x = x; vehicle.y = y;
    vehicle.yaw = yaw; vehicle.v = v; vehicle.dt = dt;
    vehicle.vpi = vpi;
    vehicle.x_front = x + vpi.L * cos(yaw);
    vehicle.y_front = y + vpi.L * sin(yaw);
    for i = 1:vpi.num_trailers
        if i == 1
            t_yaw = normalize_angle(vehicle.yaw + init_rel_angles(i));
            t_x_front = vehicle.x - vpi.Lb * cos(vehicle.yaw);
            t_y_front = vehicle.y - vpi.Lb * sin(vehicle.yaw);
            t_x = t_x_front - vpi.trailer_L(i) * cos(t_yaw);
            t_y = t_y_front - vpi.trailer_L(i) * sin(t_yaw);
            t_hitch = normalize_angle(vehicle.yaw - t_yaw);
        else
            prev = vehicle.trailer_states(i-1);
            t_yaw = normalize_angle(prev.yaw + init_rel_angles(i));
            t_x_front = prev.x - vpi.trailer_Lb(i-1) * cos(prev.yaw);
            t_y_front = prev.y - vpi.trailer_Lb(i-1) * sin(prev.yaw);
            t_x = t_x_front - vpi.trailer_L(i) * cos(t_yaw);
            t_y = t_y_front - vpi.trailer_L(i) * sin(t_yaw);
            t_hitch = normalize_angle(prev.yaw - t_yaw);
        end
        vehicle.trailer_states(i).x = t_x; vehicle.trailer_states(i).y = t_y;
        vehicle.trailer_states(i).yaw = t_yaw; vehicle.trailer_states(i).hitch_angle = t_hitch;
        vehicle.trailer_states(i).x_front = t_x_front; vehicle.trailer_states(i).y_front = t_y_front;
        vehicle.trailer_states(i).v = v; vehicle.trailer_states(i).steer = 0;
    end
end

function vehicle = update_vehicle(vehicle, a, delta)
    vpi = vehicle.vpi; dt = vehicle.dt;
    delta = max(-vpi.MAX_STEER, min(vpi.MAX_STEER, delta));
    vehicle.steer = delta;
    vehicle.v = vehicle.v + a * dt;
    vehicle.v = min(max(vehicle.v, vpi.V_MIN), vpi.V_MAX);
    vehicle.yaw = vehicle.yaw + vehicle.v / vpi.L * tan(delta) * dt;
    vehicle.yaw = normalize_angle(vehicle.yaw);
    vehicle.x = vehicle.x + vehicle.v * cos(vehicle.yaw) * dt;
    vehicle.y = vehicle.y + vehicle.v * sin(vehicle.yaw) * dt;
    vehicle.x_front = vehicle.x + vpi.L * cos(vehicle.yaw);
    vehicle.y_front = vehicle.y + vpi.L * sin(vehicle.yaw);
    
    for i = 1:vpi.num_trailers
        now_state = vehicle.trailer_states(i);
        if i == 1
            now_state.hitch_angle = normalize_angle(vehicle.yaw - now_state.yaw);
            dyaw = vehicle.v / vpi.trailer_L(i) * (sin(now_state.hitch_angle) - vpi.Lb * tan(vehicle.steer) * cos(now_state.hitch_angle) / vpi.L) * dt;
            now_state.yaw = normalize_angle(now_state.yaw + dyaw);
            now_state.v = vehicle.v * (cos(now_state.hitch_angle) + vpi.Lb * tan(vehicle.steer) * sin(now_state.hitch_angle) / vpi.L);
            now_state.x_front = vehicle.x - vpi.Lb * cos(vehicle.yaw);
            now_state.y_front = vehicle.y - vpi.Lb * sin(vehicle.yaw);
            
            hitch_err = normalize_angle(vehicle.yaw - (now_state.yaw - dyaw));
            if abs(hitch_err) < angle_deadband_for_clearance(hitch_clearance_m(), vpi.trailer_L(i)), now_state.yaw = now_state.yaw - dyaw; end
        else
            prev = vehicle.trailer_states(i-1);
            now_state.hitch_angle = normalize_angle(prev.yaw - now_state.yaw);
            dyaw = prev.v / vpi.trailer_L(i) * (sin(now_state.hitch_angle) - vpi.trailer_Lb(i-1) * tan(prev.hitch_angle) * cos(now_state.hitch_angle) / vpi.trailer_L(i-1)) * dt;
            now_state.yaw = normalize_angle(now_state.yaw + dyaw);
            now_state.v = prev.v * (cos(now_state.hitch_angle) + vpi.trailer_Lb(i-1) * tan(prev.hitch_angle) * sin(now_state.hitch_angle) / vpi.trailer_L(i-1));
            now_state.x_front = prev.x - vpi.trailer_Lb(i-1) * cos(prev.yaw);
            now_state.y_front = prev.y - vpi.trailer_Lb(i-1) * sin(prev.yaw);
            
            hitch_err = normalize_angle(prev.yaw - (now_state.yaw - dyaw));
            if abs(hitch_err) < angle_deadband_for_clearance(hitch_clearance_m(), vpi.trailer_L(i)), now_state.yaw = now_state.yaw - dyaw; end
        end
        
        now_state.x = now_state.x + now_state.v * cos(now_state.yaw) * dt;
        now_state.y = now_state.y + now_state.v * sin(now_state.yaw) * dt;
        dx = now_state.x_front - now_state.x; dy = now_state.y_front - now_state.y;
        dist = hypot(dx, dy);
        if dist > 0
            correction = dist - vpi.trailer_L(i);
            now_state.x = now_state.x + correction * (dx / dist);
            now_state.y = now_state.y + correction * (dy / dist);
            now_state.yaw = atan2(dy, dx);
        end
        vehicle.trailer_states(i) = now_state;
    end
end

function draw_vehicle_n_trailers(vehicle, ax, colors)
    vpi = vehicle.vpi;
    wheel = [-vpi.TR, vpi.TR, vpi.TR, -vpi.TR, -vpi.TR; vpi.TW/2, vpi.TW/2, -vpi.TW/2, -vpi.TW/2, vpi.TW/2];
    rot_steer = [cos(vehicle.steer), -sin(vehicle.steer); sin(vehicle.steer), cos(vehicle.steer)];
    rot_yaw = [cos(vehicle.yaw), -sin(vehicle.yaw); sin(vehicle.yaw), cos(vehicle.yaw)];
    
    fr = rot_yaw * (rot_steer * wheel + [vpi.L; -vpi.WD/2]) + [vehicle.x; vehicle.y];
    fl = rot_yaw * (rot_steer * wheel + [vpi.L; vpi.WD/2]) + [vehicle.x; vehicle.y];
    rr = rot_yaw * ([wheel(1,:); wheel(2,:) + vpi.WD/2]) + [vehicle.x; vehicle.y];
    rl = rot_yaw * ([wheel(1,:); wheel(2,:) - vpi.WD/2]) + [vehicle.x; vehicle.y];
    
    fill(ax, fr(1,:), fr(2,:), colors{1}, 'EdgeColor', 'none'); fill(ax, fl(1,:), fl(2,:), colors{1}, 'EdgeColor', 'none');
    fill(ax, rr(1,:), rr(2,:), colors{1}, 'EdgeColor', 'none'); fill(ax, rl(1,:), rl(2,:), colors{1}, 'EdgeColor', 'none');
    plot(ax, [vehicle.x+vpi.L*cos(vehicle.yaw), vehicle.x], [vehicle.y+vpi.L*sin(vehicle.yaw), vehicle.y], 'Color', colors{1}, 'LineWidth', 2);
    
    for i = 1:2:length(vehicle.trailer_states)
        color_index = mod(floor((i-1)/2) + 1, length(colors)) + 1;
        color = colors{color_index};
        
        trailer1 = vehicle.trailer_states(i);
        
        rr_wheel_t1 = wheel;
        rl_wheel_t1 = wheel;
        rr_wheel_t1(2,:) = rr_wheel_t1(2,:) + vpi.WD/2;
        rl_wheel_t1(2,:) = rl_wheel_t1(2,:) - vpi.WD/2;
        
        rot_trailer1 = [cos(trailer1.yaw), -sin(trailer1.yaw); sin(trailer1.yaw), cos(trailer1.yaw)];
        
        rr_wheel_t1 = rot_trailer1 * rr_wheel_t1 + [trailer1.x; trailer1.y];
        rl_wheel_t1 = rot_trailer1 * rl_wheel_t1 + [trailer1.x; trailer1.y];
        
        fill(ax, rr_wheel_t1(1,:), rr_wheel_t1(2,:), color, 'EdgeColor', 'none');
        fill(ax, rl_wheel_t1(1,:), rl_wheel_t1(2,:), color, 'EdgeColor', 'none');
        
        trailer1_rear_center = [trailer1.x, trailer1.y];
        rr_wheel_t1_center = mean(rr_wheel_t1(:, 1:2), 2);
        rl_wheel_t1_center = mean(rl_wheel_t1(:, 1:2), 2);
        
        plot(ax, [rr_wheel_t1_center(1), rl_wheel_t1_center(1)], [rr_wheel_t1_center(2), rl_wheel_t1_center(2)], 'Color', color, 'LineWidth', 2);
        plot(ax, trailer1.x_front, trailer1.y_front, 'o', 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', 'MarkerSize', 3);
        
        if i + 1 <= length(vehicle.trailer_states)
            trailer2 = vehicle.trailer_states(i+1);
            
            rr_wheel_t2 = wheel;
            rl_wheel_t2 = wheel;
            rr_wheel_t2(2,:) = rr_wheel_t2(2,:) + vpi.WD/2;
            rl_wheel_t2(2,:) = rl_wheel_t2(2,:) - vpi.WD/2;
            
            rot_trailer2 = [cos(trailer2.yaw), -sin(trailer2.yaw); sin(trailer2.yaw), cos(trailer2.yaw)];
            
            rr_wheel_t2 = rot_trailer2 * rr_wheel_t2 + [trailer2.x; trailer2.y];
            rl_wheel_t2 = rot_trailer2 * rl_wheel_t2 + [trailer2.x; trailer2.y];
            
            fill(ax, rr_wheel_t2(1,:), rr_wheel_t2(2,:), color, 'EdgeColor', 'none');
            fill(ax, rr_wheel_t2(1,:), rr_wheel_t2(2,:), color, 'EdgeColor', 'none');
            
            fill(ax, rr_wheel_t2(1,:), rr_wheel_t2(2,:), color, 'EdgeColor', 'none');
            fill(ax, rl_wheel_t2(1,:), rl_wheel_t2(2,:), color, 'EdgeColor', 'none');
            
            trailer2_rear_center = [trailer2.x, trailer2.y];
            rr_wheel_t2_center = mean(rr_wheel_t2(:, 1:2), 2);
            rl_wheel_t2_center = mean(rl_wheel_t2(:, 1:2), 2);
            
            plot(ax, [rr_wheel_t2_center(1), rl_wheel_t2_center(1)], [rr_wheel_t2_center(2), rl_wheel_t2_center(2)], 'Color', color, 'LineWidth', 2);
            plot(ax, [trailer1_rear_center(1), trailer2_rear_center(1)], [trailer1_rear_center(2), trailer2_rear_center(2)], 'Color', color, 'LineWidth', 2);
            
            if i == 1
                plot(ax, [vehicle.x, trailer1_rear_center(1)], [vehicle.y, trailer1_rear_center(2)], 'Color', color, 'LineWidth', 2);
            else
                prev_trailer = vehicle.trailer_states(i-1);
                plot(ax, [prev_trailer.x, trailer1_rear_center(1)], [prev_trailer.y, trailer1_rear_center(2)], 'Color', color, 'LineWidth', 2);
                plot(ax, trailer1.x_front, trailer1.y_front, 'o', 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', 'MarkerSize', 3);
            end
        else
            if i == 1
                plot(ax, [vehicle.x, trailer1_rear_center(1)], [vehicle.y, trailer1_rear_center(2)], 'Color', color, 'LineWidth', 2);
            elseif i > 1
                prev_trailer = vehicle.trailer_states(i-1);
                plot(ax, [prev_trailer.x, trailer1_rear_center(1)], [prev_trailer.y, trailer1_rear_center(2)], 'Color', color, 'LineWidth', 2);
                plot(ax, trailer1.x_front, trailer1.y_front, 'o', 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', 'MarkerSize', 3);
            end
        end
    end
end

% ==============================================================================
% 【修正版】多挂车逆向运动学轨迹生成器 (稳定版)
% 通过等效运动延展和强高斯平滑，彻底消除多阶求导带来的“高频振荡爆炸”！
% ==============================================================================
function [tr_x, tr_y, tr_yaw] = generate_inverse_kinematics_trajectory(ref_x, ref_y, vpi, target_trailer_idx)
    % target_trailer_idx: 目标挂车的索引，表示这根参考线是用来要求第几节挂车走的
    % 如果不传，默认认为是最后一节挂车 (vpi.num_trailers)
    if nargin < 4
        target_trailer_idx = vpi.num_trailers; %修改索引挂车节数（谁作为主要跟踪目标）
    end
    
    X = ref_x;
    Y = ref_y;
    
    % 只从指定的挂车序号开始，反推到牵引车 (第1节挂车再往前推一次)
    for k = target_trailer_idx:-1:1
        % 【核心修复1】：强力高斯平滑，消除因为直角向圆弧过渡时的几何突变（无限大曲率）！
        X = smoothdata(X, 'gaussian', 50);
        Y = smoothdata(Y, 'gaussian', 50);
        
        % 1. 计算当前挂车的偏航角 (沿着轨迹切线)
        YAW = zeros(size(X));
        for j = 1:(length(X)-1)
            YAW(j) = atan2(Y(j+1)-Y(j), X(j+1)-X(j));
        end
        YAW(end) = YAW(end-1);
        
        % 消除负空间相位跳变，并强制平滑角度（极其重要）
        YAW = unwrap(YAW);
        YAW = smoothdata(YAW, 'gaussian', 50);
        
        % 【核心修复2】：废弃极度不稳定且放大的曳物线微分方程(Tractrix ODE)倒推
        % 改用“等效前向延展杆”投影，这在物理上高度近似但数学上100%稳定！绝对不会震荡！
        L_k = vpi.trailer_L(k);
        if k == 1
            Lb_prev = vpi.Lb; 
        else
            Lb_prev = vpi.trailer_Lb(k-1); 
        end
        
        % 将前车的后悬与当前车的轴距视为一体直接沿切向延展
        effective_Length = L_k + Lb_prev; 
        
        X = X + effective_Length .* cos(YAW);
        Y = Y + effective_Length .* sin(YAW);
    end
    
    % 输出最前端的牵引车坐标要求前，再做最后一次平滑保证 NMPC 收敛极快
    tr_x = smoothdata(X, 'gaussian', 60);
    tr_y = smoothdata(Y, 'gaussian', 60);
    
    % 算出牵引车在这个新轨迹上应该具备的偏航角
    tr_yaw = zeros(size(tr_x));
    for j = 1:(length(tr_x)-1)
        tr_yaw(j) = atan2(tr_y(j+1)-tr_y(j), tr_x(j+1)-tr_x(j));
    end
    tr_yaw(end) = tr_yaw(end-1);
    
    % 【关键】：由于基于离散点计算的 atan2 极其容易出现数值跳变和毛刺，
    % 这直接导致了牵引车 NMPC 在追踪角度时“疯狂画龙（Shake/Oscillate）”
    % 所以最后必须要解包裹并进行超级平滑！
    tr_yaw = unwrap(tr_yaw);
    tr_yaw = smoothdata(tr_yaw, 'gaussian', 100);
end
