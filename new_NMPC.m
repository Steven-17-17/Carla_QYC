%% 工业牵引车 + 多节挂车 NMPC 路径跟踪 (转弯工况 + 完整可视化)
load("plane_obstacle_edge.mat")
map_res = 4; % 分辨率：每米 2 个网格 (精度 0.5 米)

% 根据多边形的最值，向外扩展 20 米作为地图边界
map_x_min = min(physical_x_centered) - 20;
map_x_max = max(physical_x_centered) + 20;
map_y_min = min(physical_y_centered) - 20;
map_y_max = max(physical_y_centered) + 20;

% 创建网格坐标点阵
x_vec = map_x_min : (1/map_res) : map_x_max;
y_vec = map_y_min : (1/map_res) : map_y_max;
[X_grid, Y_grid] = meshgrid(x_vec, y_vec);

% 使用 inpolygon 将物理多边形转化为二值矩阵
% inPlane 中，在多边形内部的网格值为 1，外部为 0
inPlane = inpolygon(X_grid, Y_grid, physical_x_centered, physical_y_centered);

% 转换为 double 类型，适配 Coder 和寻路算法
grid_map = double(inPlane);

%% 障碍物膨胀 (安全边界设计)
% 牵引车的半宽大约是 1~1.5 米。如果不膨胀，算法规划的路径会让车辆中心点贴着机翼走
safe_distance_m = 0.5; 
safe_grids = round(safe_distance_m * map_res); % 转换为网格数

% 使用形态学膨胀，把飞机轮廓“胖”一圈
se = strel('disk', safe_grids);
grid_map = imdilate(grid_map, se);

%% 1. 核心参数定义
params.N  = 8;                % 挂车数量
params.l1 = 2.0;              % 牵引车轴距
params.l  = [0, 1.4, 2.0, 1.4, 2.0, 1.4, 2.0, 1.4, 2.0]; % 各节长度
% params.e  = [0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0];    % 离轴距离
params.e  = [1.15, 0, 1.15, 0, 1.15, 0, 1.15, 0];    % 正确的离轴距离
W_veh = 2.0; L_truck = 3.3; L_trail = 3.3;
wheel_r = 0.35; wheel_W = 0.2;

%% 2. 参考路径生成
% R_curve = 50; 
% ds = 0.2; % 路点间距为 0.2m
% 
% % 弯道前的直线
% ref_x_pre = -30:ds:0;
% ref_y_pre = zeros(size(ref_x_pre));
% ref_yaw_pre = zeros(size(ref_x_pre));
% 
% % 90度圆弧
% d_theta = ds / R_curve; 
% theta_ref = d_theta : d_theta : (pi/2);
% ref_x_arc = R_curve * sin(theta_ref);
% ref_y_arc = R_curve * (1 - cos(theta_ref));
% ref_yaw_arc = theta_ref;
% 
% % 弯道后的直线
% ref_y_post = (ref_y_arc(end) + ds) : ds : (ref_y_arc(end) + 40);
% ref_x_post = ones(size(ref_y_post)) * ref_x_arc(end);
% ref_yaw_post = ones(size(ref_y_post)) * (pi/2);
% 
% ref_x = [ref_x_pre, ref_x_arc, ref_x_post];
% ref_y = [ref_y_pre, ref_y_arc, ref_y_post];
% ref_yaw = [ref_yaw_pre, ref_yaw_arc, ref_yaw_post];
% refPath = [ref_x', ref_y', ref_yaw'];
%% 2. 参考路径生成 (S型双圆弧轨迹 - 曲率积分法)
% 使用曲率积分法保证所有轨迹点严格等距 (ds = 0.2)
% 这对于“动态时间重采样”的 NMPC 逻辑至关重要

% ds = 0.2;         % 严格等距
% R_curve = 10;     % 转弯半径 (8节挂车极长，建议 R 设在 40-50 左右)
% kappa_max = 1 / R_curve; 
% 
% % 1. 定义各段轨迹长度
% L_straight1 = 30;              % 起步直线段长度
% L_arc1 = R_curve * (pi/3);     % 第一段弧：向左转 60度 (\pi/3)
% L_arc2 = R_curve * (pi/3);     % 第二段弧：向右转 60度 回正
% L_straight2 = 60;              % 弯后直线段长度
% 
% % 2. 计算各段对应的离散点数
% N_s1 = round(L_straight1 / ds);
% N_a1 = round(L_arc1 / ds);
% N_a2 = round(L_arc2 / ds);
% N_s2 = round(L_straight2 / ds);
% 
% % 3. 构建完整的曲率数组 (Kappa)
% % 正曲率 = 左转，负曲率 = 右转，0 = 直行
% kappa = [zeros(1, N_s1), kappa_max * ones(1, N_a1), -kappa_max * ones(1, N_a2), zeros(1, N_s2)];
% 
% % 4. 预分配空间
% total_pts = length(kappa);
% ref_x = zeros(1, total_pts);
% ref_y = zeros(1, total_pts);
% ref_yaw = zeros(1, total_pts);
% 
% % 5. 设定起点坐标
% ref_x(1) = -30;
% ref_y(1) = 0;
% ref_yaw(1) = 0;
% 
% % 6. 运动学积分生成坐标 (保证绝对 C1 连续和完美等距)
% for i = 2:total_pts
%     ref_yaw(i) = ref_yaw(i-1) + kappa(i) * ds;
%     ref_x(i) = ref_x(i-1) + cos(ref_yaw(i)) * ds;
%     ref_y(i) = ref_y(i-1) + sin(ref_yaw(i)) * ds;
% end
% 
% % 组合成最终的全局参考路径
% refPath = [ref_x', ref_y', ref_yaw'];
load("hybrid_astar_path_0331.mat")
refPath=path;
ref_x = path(:,1)';
ref_y = path(:,2)';
ref_yaw = path(:,3)';
%% 3. 仿真与 NMPC 配置
dt = 0.2; 
N_p = 15; % 预测时域
x_curr = [ref_x(1), ref_y(1), -pi/2, -pi/2, -pi/2, -pi/2, -pi/2,-pi/2,-pi/2,-pi/2,-pi/2, 0.5]; % 初始状态
% x_curr = [ref_x(1), ref_y(1), zeros(1, params.N+1), 1];
% 重新计算仿真总步数
T_sim = 5000;

%% 4. 绘图初始化
figure('Color','w','Position',[100,100,1000,700]);
axis equal; grid on; hold on;
% =======================================================
% --- 新增：绘制飞机（静态障碍物）轮廓 ---
% 使用 fill 函数绘制一个半透明的红色多边形代表禁行区
fill(physical_x_centered, physical_y_centered, 'r', ...
    'FaceAlpha', 0.2, 'EdgeColor', '#CC0000', 'LineWidth', 1.5);
% =======================================================
plot(ref_x, ref_y, 'k--', 'LineWidth', 1.5); % 绘制参考路径
h_links = plot(0,0,'k-','LineWidth',1.2);
h_axle_dots = plot(0,0,'ko','MarkerFaceColor','w','MarkerSize',5);
h_hitch_dots = plot(0,0,'ko','MarkerFaceColor','y','MarkerSize',6);

% 创建车体句柄
num_bodies = 1 + ceil(params.N/2);
h_bodies = gobjects(num_bodies, 1);
for i=1:num_bodies, h_bodies(i) = plot(0,0,'LineWidth',2); end
set(h_bodies(1), 'Color', 'b'); 

% 轴与轮子句柄
num_axles = 1 + (params.N + 1);
h_axles = gobjects(num_axles, 1);
h_wheels = gobjects(num_axles * 2, 1);
for i=1:num_axles, h_axles(i) = plot(0,0,'k-','LineWidth',2); end
for i=1:(num_axles*2), h_wheels(i) = plot(0,0,'Color',[0.2 0.2 0.2],'LineWidth',1.2); end

% --- 视频录制初始化 ---
record_video = false; % 设置为 true 开启视频录制，false 则只显示动画
if record_video
    video_filename = 'TractorTrailer_NMPC_Tracking.mp4';
    v_writer = VideoWriter(video_filename, 'MPEG-4');
    % 将视频帧率设置为 1/dt (这里 dt=0.2，所以帧率为 5 FPS)
    % 这样视频播放的速度就与真实的物理时间流逝速度完全一致
    v_writer.FrameRate = 1 / dt; 
    v_writer.Quality = 100; % 最高画质
    open(v_writer);
end

%% 5. 主仿真循环
k = 1;
max_steps = 2000; 
dist_to_end = inf;
% --- 新增：数据记录初始化 ---
history_x = [];   % 记录所有状态量
history_cte = []; % 记录横向跟踪误差
history_t = [];   % 记录时间
t_current = 0;    % 当前仿真时间
% ---------------------------
while dist_to_end > 1.0 && k < T_sim
    % --- 修改为主循环代码 ---
    u_cmd = nmpc_solver(x_curr, refPath, params, N_p, dt);
    delta_cmd = u_cmd(1);
    v_cmd     = u_cmd(2);
    
    % 更新当前状态的速度项，喂给积分器
    x_curr(end) = v_cmd; 
    
    % 此时 ode45 里的 kinematics 函数读取的 x(end) 就是最新的 v_cmd
    [~, X_all] = ode45(@(t,x) tractor_trailer_kinematics(x, delta_cmd, params), [0 dt], x_curr);
    x_curr = X_all(end, :);
    % --- 新增：记录状态与计算跟踪误差 ---
    history_x = [history_x; x_curr];
    history_t = [history_t; t_current];
    t_current = t_current + dt;
    
    % 计算当前前轴的实际横向误差 (CTE)
    P_f0_x = x_curr(1) + params.l1 * cos(x_curr(3));
    P_f0_y = x_curr(2) + params.l1 * sin(x_curr(3));
    dist_sq = (refPath(:,1) - P_f0_x).^2 + (refPath(:,2) - P_f0_y).^2;
    [~, current_idx] = min(dist_sq);
    ref_p = refPath(current_idx, :);
    dx_err = P_f0_x - ref_p(1);
    dy_err = P_f0_y - ref_p(2);
    % 横向误差公式：沿着参考点法线方向的投影
    cte = dx_err * (-sin(ref_p(3))) + dy_err * cos(ref_p(3));
    history_cte = [history_cte; cte];
    % ------------------------------------
    % 更新距离终点的距离
    dist_to_end = norm(x_curr(1:2) - refPath(end, 1:2));
    k = k + 1;

    % --- 提取坐标并绘图 ---
    xr1 = x_curr(1); yr1 = x_curr(2); th = x_curr(3:3+params.N);
    P = zeros(params.N+1, 2); H = zeros(params.N, 2); P(1,:) = [xr1, yr1];
    for i = 1:params.N
        H(i, :) = P(i, :) - params.e(i) * [cos(th(i)), sin(th(i))];
        P(i+1, :) = H(i, :) - params.l(i+1) * [cos(th(i+1)), sin(th(i+1))];
    end
    P_f = P(1,:) + [params.l1*cos(th(1)), params.l1*sin(th(1))];
    
    % 更新可视化元素
    % 1. 骨架与点
    spine_pts = [P_f; P(1,:)];
    for i=1:params.N, spine_pts = [spine_pts; H(i,:); P(i+1,:)]; end
    set(h_links, 'XData', spine_pts(:,1), 'YData', spine_pts(:,2));
    set(h_axle_dots, 'XData', [P_f(1); P(:,1)], 'YData', [P_f(2); P(:,2)]);
    set(h_hitch_dots, 'XData', H(:,1), 'YData', H(:,2));
    
    % 2. 轮轴 (前轴转向 delta_cmd)
    [w_f, ax_f] = get_axle_elements(P_f(1), P_f(2), th(1), W_veh, wheel_r, wheel_W, delta_cmd);
    set(h_axles(1), 'XData', ax_f(:,1), 'YData', ax_f(:,2));
    set(h_wheels(1), 'XData', w_f{1}(:,1), 'YData', w_f{1}(:,2));
    set(h_wheels(2), 'XData', w_f{2}(:,1), 'YData', w_f{2}(:,2));
    for i = 1:(params.N+1)
        [w_i, ax_i] = get_axle_elements(P(i,1), P(i,2), th(i), W_veh, wheel_r, wheel_W, 0);
        set(h_axles(i+1), 'XData', ax_i(:,1), 'YData', ax_i(:,2));
        set(h_wheels(2*i+1), 'XData', w_i{1}(:,1), 'YData', w_i{1}(:,2));
        set(h_wheels(2*i+2), 'XData', w_i{2}(:,1), 'YData', w_i{2}(:,2));
    end
    
    % 3. 车体
    draw_body(h_bodies(1), P_f, P(1,:), th(1), L_truck, W_veh);
    b_idx = 2;
    for i = 2:2:params.N
        if b_idx <= num_bodies
            draw_body(h_bodies(b_idx), P(i,:), P(i+1,:), th(i+1), L_trail, W_veh);
            b_idx = b_idx + 1;
        end
    end
    
    % 相机跟随
    axis(gca, [xr1-30, xr1+30, yr1-20, yr1+20]);
    % 为了保证视频录制不漏帧，将 drawnow limitrate 改为 drawnow
    drawnow; 
    
    % --- 写入视频帧 ---
    if record_video
        frame = getframe(gcf);
        writeVideo(v_writer, frame);
    end
end
if record_video
    close(v_writer); % 必须执行这一步，文件才会生成索引
    fprintf('视频保存成功: %s\n', video_filename);
end

% ==========================================================
% --- 新增：绘制航向角与跟踪误差曲线 ---
figure('Color','w','Position',[200, 100, 800, 700], 'Name', 'Data Analysis');

% 1. 绘制各车体航向角
subplot(2, 1, 1);
hold on; grid on;
% 牵引车的航向角是第 3 个状态量
plot(history_t, history_x(:, 3), 'b-', 'LineWidth', 2, 'DisplayName', 'Tractor'); 
% 绘制 8 节挂车的航向角 (状态量 4 到 11)
colors = lines(params.N); % 获取默认配色
for i = 1:params.N
    plot(history_t, history_x(:, 3+i), '-', 'Color', colors(i,:), ...
        'LineWidth', 1.2, 'DisplayName', sprintf('Trailer %d', i));
end
xlabel('Time (s)', 'FontSize', 11);
ylabel('Heading Angle (rad)', 'FontSize', 11);
title('Heading Angles of Tractor and Trailers', 'FontSize', 12);
legend('Location', 'best', 'NumColumns', 3);

% 2. 绘制牵引车前轴的跟踪误差 (CTE)
subplot(2, 1, 2);
plot(history_t, history_cte, 'r-', 'LineWidth', 2);
grid on;
xlabel('Time (s)', 'FontSize', 11);
ylabel('Cross Track Error (m)', 'FontSize', 11);
title('Tracking Error (Tractor Front Axle)', 'FontSize', 12);
% ==========================================================
%% NMPC 核心函数
%% 优化后的 NMPC 核心函数 (双输入 + 精准弧长跟踪)
function u_opt = nmpc_solver(x0, refPath, params, N_p, dt)
    persistent last_u_seq; 
    
    % u_seq 现在是 N_p x 2 的矩阵：[转角 delta, 速度 v]
    if isempty(last_u_seq)
        last_u_seq = zeros(N_p, 2); 
        last_u_seq(:, 2) = x0(end); % 初始化速度为当前状态速度
    end
    
    % 热启动：用上一帧的解向后平移
    u0 = [last_u_seq(2:end, :); last_u_seq(end, :)]; 
    
    % 展开成 1D 向量喂给 fmincon: [delta_1, v_1, delta_2, v_2, ..., delta_Np, v_Np]
    u0_vec = reshape(u0', [], 1); 
    
    % 设置控制量的物理边界[cite: 2]
    max_steer = 0.2; % 最大转角 (rad)
    min_v = 0.0;     % 最小速度 (m/s) - 允许停车
    max_v = 1.0;     % 最大速度 (m/s)
    
    lb_vec = repmat([-max_steer; min_v], N_p, 1);
    ub_vec = repmat([max_steer; max_v], N_p, 1);
    
    opts = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp', ...
                        'MaxIterations', 50, 'StepTolerance', 1e-4);
                    
    [u_seq_vec, ~, exitflag] = fmincon(@(u) nmpc_cost(u, x0, refPath, params, N_p, dt), ...
                                       u0_vec, [],[],[],[], lb_vec, ub_vec, [], opts);
    
    % 恢复为 N_p x 2 的矩阵
    u_seq_mat = reshape(u_seq_vec, 2, [])'; 
    
    % 求解保护
    if exitflag <= 0
        u_opt = last_u_seq(2, :); % 求解失败，使用上一帧的安全指令
    else
        u_opt = u_seq_mat(1, :);  % 提取当前步的 [delta, v]
        last_u_seq = u_seq_mat; 
    end
end

function J = nmpc_cost(u_vec, x0, refPath, params, N_p, dt)
    J = 0; 
    xt = reshape(x0, 1, []); % 当前状态行向量
    u = reshape(u_vec, 2, [])'; % 恢复为 N_p x 2: [delta, v]
    
    % 1. 预计算参考路径的累计弧长 (极其关键的改进)
    % 假设 refPath 已经是等距采样 (ds = 0.2)
    ds = 0.5; 
    
    % 找到当前车辆在路径上的最近点
    P_f0_x = xt(1) + params.l1 * cos(xt(3));
    P_f0_y = xt(2) + params.l1 * sin(xt(3));
    dist_sq = (refPath(:,1) - P_f0_x).^2 + (refPath(:,2) - P_f0_y).^2;
    [~, current_idx] = min(dist_sq);
    
    % 初始化预测步内的累计行驶距离
    accumulated_s = 0; 
    
    for i = 1:N_p
        delta_cmd = u(i, 1);
        v_cmd     = u(i, 2);
        
        % 临时将状态向量最后一位的速度替换为当前的指令速度，以便动力学计算[cite: 2]
        xt(end) = v_cmd; 
        
        % 2. 系统预测更新 (前向积分)
        dx_state = tractor_trailer_kinematics(xt, delta_cmd, params);
        xt = xt + reshape(dx_state, 1, []) * dt; 
        
        % 3. 精准匹配参考点：基于预测行驶弧长
        accumulated_s = accumulated_s + v_cmd * dt; % 预测车辆走过的真实弧长
        idx_offset = round(accumulated_s / ds);     % 换算为参考点的索引偏移量
        
        target_idx = current_idx + idx_offset;
        target_idx = min(target_idx, size(refPath, 1)); % 防止越界
        ref_p = refPath(target_idx, :);
        
        % 4. 计算前轴位置与横向误差
        P_f = [xt(1) + params.l1*cos(xt(3)), xt(2) + params.l1*sin(xt(3))];
        
        dx_err = P_f(1) - ref_p(1);
        dy_err = P_f(2) - ref_p(2);
        cte = dx_err * (-sin(ref_p(3))) + dy_err * cos(ref_p(3));
        
        yaw_err = mod(xt(3) - ref_p(3) + pi, 2*pi) - pi;
        
        % 5. 动态惩罚权重设计
        W_cte = 10;    % 加大横向误差惩罚，防切内角
        W_yaw = 20;    % 允许稍微偏头
        
        J = J + W_cte * cte^2 + W_yaw * yaw_err^2;
        
        % 6. 控制增量与速度偏好惩罚
        W_du_steer = 30; % 惩罚急打方向
        W_du_v = 10;     % 惩罚急刹/急加速
        W_v_ref = 2;     % 鼓励保持较高的巡航速度，防止 NMPC 偷懒停车
        
        v_ref = 1.5;     % 期望巡航速度 0.5m/s
        J = J + W_v_ref * (v_cmd - v_ref)^2; 
        
        if i > 1
            J = J + W_du_steer * (u(i, 1) - u(i-1, 1))^2; 
            J = J + W_du_v     * (u(i, 2) - u(i-1, 2))^2; 
        else
            % 惩罚与第一步的绝对输入过大
            J = J + 50 * u(i, 1)^2; 
        end
    end
end


%% 辅助函数 

function [w, axle_line] = get_axle_elements(xc, yc, theta, W, r, ww, delta)
    % 如果没有传入 delta，默认为 0（用于后轴）
    if nargin < 7, delta = 0; end 
    % 1. 计算轴的两个端点位置 (始终垂直于车身 theta)
    % 这里的 theta 是车身朝向，轴方向应为 theta + pi/2
    vec_axle = [-sin(theta), cos(theta)] * (W/2);
    L_pt = [xc, yc] + vec_axle; 
    R_pt = [xc, yc] - vec_axle;
    axle_line = [L_pt; R_pt]; % 轴线不随 delta 旋转
    
    % 2. 计算轮子的旋转矩阵
    % 后轮 delta=0，则随车身 theta 转；前轮则随 theta + delta 转
    total_wheel_angle = theta + delta;
    R_mat = [cos(total_wheel_angle), -sin(total_wheel_angle); 
             sin(total_wheel_angle), cos(total_wheel_angle)];
    
    % 轮子本地坐标 (矩形)
    wb = [-r, -ww/2; r, -ww/2; r, ww/2; -r, ww/2; -r, -ww/2];
    
    % 3. 将轮子放置到轴端点上
    w{1} = L_pt + wb * R_mat'; 
    w{2} = R_pt + wb * R_mat';
end

function draw_body(h, P_front, P_rear, theta, L, W)
    center = (P_front + P_rear) / 2;
    pts = [-L/2, -W/2; L/2, -W/2; L/2, W/2; -L/2, W/2; -L/2, -W/2];
    R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    pts = center + pts * R';
    set(h, 'XData', pts(:,1), 'YData', pts(:,2));
end


function dx = tractor_trailer_kinematics(x, delta, params)
    % 内部统一使用列向量计算，输出时适配输入形状
    x_col = x(:); 
    N = params.N; l1 = params.l1; l = params.l; e = params.e;
    theta = x_col(3:3+N); v = x_col(end);
    
    dx_col = zeros(size(x_col));
    dx_col(1) = v * cos(theta(1)); 
    dx_col(2) = v * sin(theta(1));
    
    dth = zeros(N+1, 1); 
    dth(1) = v * tan(delta) / l1;
    
    cv = v;
    for i = 1:N
        dth(i+1) = (cv*sin(theta(i)-theta(i+1)) - e(i)*cos(theta(i)-theta(i+1))*dth(i)) / l(i+1);
        cv = cv*cos(theta(i)-theta(i+1)) + e(i)*sin(theta(i)-theta(i+1))*dth(i);
    end
    dx_col(3:3+N) = dth;
    
    % 恢复输入时的向量形状
    if isrow(x)
        dx = dx_col';
    else
        dx = dx_col;
    end
end
