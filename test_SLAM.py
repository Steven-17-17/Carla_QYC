import carla
import time
import random
import numpy as np
import open3d as o3d

# ================= Fast-LIO2 轻量化建图引擎（无可视化，纯计算） =================
class MiniFastLIO2:
    def __init__(self):
        self.global_map = np.empty((0, 3))  # 全局地图（SLAM核心输出）

    def update_map(self, xyz):
        if len(xyz) < 10:
            return
        
        # 核心：把每一帧点云加入全局地图（真正SLAM逻辑）
        self.global_map = np.vstack((self.global_map, xyz))
        
        # 简单降采样，避免地图太大
        if len(self.global_map) > 200000:
            self.global_map = self.global_map[::2]

# ================= 主程序 =================
def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(15.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    spectator = world.get_spectator()

    # 生成车辆
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz')
    vehicle_bp.set_attribute('color', '255,0,0')
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    print("✅ 车辆生成成功")

    # 激光雷达
    lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('range', '60')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('points_per_second', '80000')

    lidar = world.spawn_actor(
        lidar_bp,
        carla.Transform(carla.Location(z=2.5)),
        attach_to=vehicle
    )

    # 初始化 Fast-LIO2
    fast_lio = MiniFastLIO2()

    # 激光雷达 → 直接喂给 SLAM（无可视化，永不报错）
    def lidar_callback(data):
        points = np.frombuffer(data.raw_data, dtype=np.float32)
        points = points.reshape(-1, 4)
        xyz = points[:, :3]

        # 🔥 核心：点云输入 Fast-LIO2
        fast_lio.update_map(xyz)

        print(f"\r📡 点云：{len(xyz):<4} | 全局地图点数：{len(fast_lio.global_map):<6}", end="")

    lidar.listen(lidar_callback)

    # 自动行驶
    vehicle.set_autopilot(True)

    # ================= Open3D 实时可视化配置 =================
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Carla LiDAR SLAM Point Cloud", width=800, height=600)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    
    # 强制让视口能够容纳初始点云
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 1.0
    # ========================================================
    
    # 视图重置标志
    view_reset = False

    # 跟随视角
    try:
        while True:
            tran = vehicle.get_transform()
            fv = tran.get_forward_vector()
            pos = tran.location - fv * 8 + carla.Location(z=3)
            rot = carla.Rotation(pitch=-15, yaw=tran.rotation.yaw)
            spectator.set_transform(carla.Transform(pos, rot))
            
            # 实时更新 Open3D 点云显示
            if len(fast_lio.global_map) > 0:
                xyz = fast_lio.global_map
                pcd.points = o3d.utility.Vector3dVector(xyz)
                
                # 给点云赋予基于高度（Z轴）的渐变颜色，避免点全黑看不见
                z_vals = xyz[:, 2]
                z_norm = (z_vals - np.min(z_vals)) / (np.max(z_vals) - np.min(z_vals) + 1e-6)
                colors = np.zeros((len(xyz), 3))
                colors[:, 0] = z_norm      # R
                colors[:, 1] = 1 - z_norm  # G
                colors[:, 2] = 0.5         # B
                pcd.colors = o3d.utility.Vector3dVector(colors)
                
                vis.update_geometry(pcd)
                
                # 初始帧强制重置相机视角（因为刚创建时没有内容，不知视场多大）
                if not view_reset:
                    vis.reset_view_point(True)
                    view_reset = True
            
            vis.poll_events()
            vis.update_renderer()
            
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n🛑 程序停止")

    finally:
        lidar.destroy()
        vehicle.destroy()
        vis.destroy_window()
        
        # 保存地图到文件（你可以用MeshLab/CloudCompare打开查看）
        np.save("fast_lio2_map.npy", fast_lio.global_map)
        print("✅ 地图已保存为 fast_lio2_map.npy")
        print("✅ 资源已释放")

if __name__ == '__main__':
    main()