% change_road_side
% 读取并显示 lattice 边界点（参考线、左边界、右边界）

clear; clc; close all;

% 优先在脚本所在目录找文件，找不到再在当前工作目录找
scriptPath = mfilename('fullpath');
if isempty(scriptPath)
	baseDir = pwd;
else
	baseDir = fileparts(scriptPath);
end

candidateFiles = {
	'lattice_coundaries_all.csv', ...
	'lattice_coundaries_all', ...
	'lattice_boundaries_all.csv', ...
	'lattice_boundaries_all'
};

csvPath = '';
for i = 1:numel(candidateFiles)
	p1 = fullfile(baseDir, candidateFiles{i});
	p2 = fullfile(pwd, candidateFiles{i});
	if exist(p1, 'file') == 2
		csvPath = p1;
		break;
	elseif exist(p2, 'file') == 2
		csvPath = p2;
		break;
	end
end

if isempty(csvPath)
	error('未找到边界文件，请确认 lattice_boundaries_all.csv 在脚本目录或当前工作目录。');
end

opts = detectImportOptions(csvPath, 'VariableNamingRule', 'preserve');
T = readtable(csvPath, opts);

requiredCols = {
	'idx', 'ref_x', 'ref_y', 'road_id', 'section_id', 'lane_id', ...
	'lane_width', 'left_drivable_width', 'right_drivable_width', ...
	'left_boundary_x', 'left_boundary_y', 'right_boundary_x', 'right_boundary_y'
};

missingCols = requiredCols(~ismember(requiredCols, T.Properties.VariableNames));
if ~isempty(missingCols)
	error('CSV 缺少字段: %s', strjoin(missingCols, ', '));
end

fprintf('已读取: %s\n', csvPath);
fprintf('点数: %d\n', height(T));

figure('Color', 'w', 'Name', 'Lattice Boundaries Viewer');
hold on;
grid on;
axis equal;

% 画点（满足你“显示点”的需求）
scatter(T.ref_x, T.ref_y, 10, [0.2 0.7 0.2], 'filled', 'DisplayName', 'ref points');
scatter(T.left_boundary_x, T.left_boundary_y, 10, [0.1 0.2 0.9], 'filled', 'DisplayName', 'left boundary points');
scatter(T.right_boundary_x, T.right_boundary_y, 10, [0.1 0.7 0.7], 'filled', 'DisplayName', 'right boundary points');

% 同时画线，便于观察连续性
plot(T.ref_x, T.ref_y, '--', 'Color', [0.2 0.7 0.2], 'LineWidth', 1.0, 'HandleVisibility', 'off');
plot(T.left_boundary_x, T.left_boundary_y, '-', 'Color', [0.1 0.2 0.9], 'LineWidth', 1.2, 'HandleVisibility', 'off');
plot(T.right_boundary_x, T.right_boundary_y, '-', 'Color', [0.1 0.7 0.7], 'LineWidth', 1.2, 'HandleVisibility', 'off');

xlabel('X (m)');
ylabel('Y (m)');
title('Lattice Boundaries Points');
legend('Location', 'best');

% 可选：标注少量 idx（防止太密）
step = max(1, floor(height(T) / 40));
sampleIdx = 1:step:height(T);
text(T.ref_x(sampleIdx), T.ref_y(sampleIdx), string(T.idx(sampleIdx)), ...
	'FontSize', 7, 'Color', [0.1 0.5 0.1], 'HorizontalAlignment', 'left');

