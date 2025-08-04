
import numpy as np
from pxr import Usd, UsdGeom, Vt, Gf
import open3d as o3d

def pointcloud_to_usd_cubes(input_file, output_file, cube_size=0.1):
    """
    将点云文件转换为正方体USD文件
    
    Args:
        input_file: 输入文件路径 (.pcd 或 .e57)
        output_file: 输出USD文件路径
        cube_size: 正方体大小
    """
    
    # 读取点云
    print(f"读取点云: {input_file}")
    pcd = o3d.io.read_point_cloud(input_file)
    
    if len(pcd.points) == 0:
        print("错误: 无法读取点云文件")
        return
    
    # 获取点坐标
    points = np.asarray(pcd.points)
    
    # 创建USD文件
    stage = Usd.Stage.CreateNew(output_file)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    
    # 创建根节点
    root_xform = UsdGeom.Xform.Define(stage, "/PointCloudCubes")
    
    print(f"创建 {len(points)} 个正方体...")
    
    # 为每个点创建一个正方体
    for i, point in enumerate(points):
        if i % 1000 == 0:
            print(f"处理进度: {i}/{len(points)}")
        
        # 创建正方体
        cube_path = f"/PointCloudCubes/Cube_{i}"
        cube_prim = UsdGeom.Cube.Define(stage, cube_path)
        
        # 设置正方体大小
        cube_prim.GetSizeAttr().Set(cube_size)
        
        # 设置正方体位置
        cube_prim.AddTranslateOp().Set(Gf.Vec3f(point[0], point[1], point[2]))
        
        # 设置灰色材质
        cube_prim.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.0, 0.0)]))
    
    # 保存文件
    stage.Save()
    print(f"转换完成: {output_file}")
    print(f"生成了 {len(points)} 个正方体")


if __name__ == "__main__":

    input_file = "/home/hu/caric_docker/ws_caric/src/caric_mission/models/crane/crane_interest_points.pcd"
    output_file = "/home/hu/CaricRL/CoverageRL/isaac-training/training/models/crane/crane_interest_points.usd"
    
    pointcloud_to_usd_cubes(input_file, output_file, cube_size=0.1)

