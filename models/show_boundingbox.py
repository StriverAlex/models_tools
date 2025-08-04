
"""python show_boundingbox.py "/home/hu/CaricRL/CoverageRL/isaac-training/training/models/hangar/airplane.usd" --show-bounding-boxes --bbox-mode aabb"""
import argparse
import math

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to view USD models in Isaac Sim.")
parser.add_argument("input", type=str, help="The path to the input USD file to view.")
parser.add_argument(
    "--physics",
    action="store_true",
    default=False,
    help="Enable physics simulation.",
)
parser.add_argument(
    "--gravity",
    type=float,
    default=-9.81,
    help="Set gravity value (default: -9.81).",
)
parser.add_argument(
    "--show-collision",
    action="store_true",
    default=False,
    help="Show collision meshes.",
)
parser.add_argument(
    "--camera-distance",
    type=float,
    default=5.0,
    help="Set camera distance from model (default: 5.0).",
)
parser.add_argument(
    "--no-ground-plane",
    action="store_true",
    default=False,
    help="Disable ground plane.",
)
parser.add_argument(
    "--show-bounding-boxes",
    action="store_true",
    default=False,
    help="Show bounding boxes for model components.",
)
parser.add_argument(
    "--bbox-mode",
    type=str,
    choices=["aabb", "obb", "both"],
    default="obb",
    help="Bounding box type to display (default: obb).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os
import numpy as np

import carb
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.utils.bounds import (
    compute_aabb, 
    compute_obb, 
    compute_combined_aabb,
    create_bbox_cache,
    get_obb_corners
)
import omni.kit.app
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics, Sdf

from omni.isaac.orbit.utils.assets import check_file_path
from omni.isaac.orbit.sim import SimulationContext


def setup_camera(stage, distance: float = 5.0):
    """Setup camera to view the model."""
    # Get the default camera
    camera_prim = stage.GetPrimAtPath("/OmniverseKit_Persp")
    if camera_prim:
        # Set camera position
        camera_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(distance, distance, distance))
        camera_prim.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3d(-25, 45, 0))


def add_ground_plane(stage):
    """Add a ground plane to the scene."""
    # Create ground plane
    ground_prim = UsdGeom.Mesh.Define(stage, "/World/GroundPlane")
    ground_prim.CreatePointsAttr([(-10, -10, 0), (10, -10, 0), (10, 10, 0), (-10, 10, 0)])
    ground_prim.CreateNormalsAttr([(0, 0, 1)] * 4)
    ground_prim.CreateFaceVertexCountsAttr([4])
    ground_prim.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    
    # Add material
    ground_prim.GetPrim().GetAttribute("primvars:displayColor").Set([(0.5, 0.5, 0.5)])
    
    # Add collision if physics is enabled
    if args_cli.physics:
        UsdPhysics.CollisionAPI.Apply(ground_prim.GetPrim())


def setup_physics(stage):
    """Setup physics scene."""
    # Create physics scene
    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr().Set(abs(args_cli.gravity))


def show_collision_meshes(stage):
    """Show collision meshes in the scene."""
    # Find all collision meshes and make them visible
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            # Make collision mesh visible
            prim.GetAttribute("visibility").Set("inherited")
            # Set a different color for collision meshes
            if prim.GetAttribute("primvars:displayColor"):
                prim.GetAttribute("primvars:displayColor").Set([(1.0, 0.0, 0.0)])  # Red color


def create_wireframe_cube(stage, corners, name, color=(1.0, 0.0, 0.0)):
    """Create a wireframe cube from 8 corner points."""
    # Define the edges of a cube (connecting corner indices)
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # bottom face
        (4, 5), (5, 7), (7, 6), (6, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
    ]
    
    # Create line segments for each edge
    line_points = []
    for edge in edges:
        line_points.extend([corners[edge[0]], corners[edge[1]]])
    
    # Create a curve prim for the wireframe
    curve_prim = UsdGeom.BasisCurves.Define(stage, f"/World/BoundingBoxes/{name}")
    curve_prim.CreatePointsAttr(line_points)
    
    # Set curve properties
    vertex_counts = [2] * len(edges)  # Each edge has 2 vertices
    curve_prim.CreateCurveVertexCountsAttr(vertex_counts)
    curve_prim.CreateTypeAttr("linear")
    curve_prim.CreateBasisAttr("bezier")
    
    # Set color
    curve_prim.GetPrim().GetAttribute("primvars:displayColor").Set([color])
    
    # Set line width
    curve_prim.CreateWidthsAttr([0.1] * len(line_points))
    
    return curve_prim


def create_aabb_wireframe(stage, aabb, name, color=(0.0, 1.0, 0.0)):
    """Create wireframe for AABB."""
    # Convert AABB to 8 corners
    min_x, min_y, min_z, max_x, max_y, max_z = aabb
    corners = [
        (min_x, min_y, min_z), (min_x, min_y, max_z),
        (min_x, max_y, min_z), (min_x, max_y, max_z),
        (max_x, min_y, min_z), (max_x, min_y, max_z),
        (max_x, max_y, min_z), (max_x, max_y, max_z)
    ]
    
    return create_wireframe_cube(stage, corners, f"{name}_AABB", color)


def analyze_model_components(stage, exclude_ground=True):
    """Analyze model components and identify main parts."""
    cache = create_bbox_cache()
    components = []
    
    # Find all mesh prims
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            prim_path = str(prim.GetPath())
            
            # Skip ground plane if requested
            if exclude_ground and "GroundPlane" in prim_path:
                continue
                
            try:
                aabb = compute_aabb(cache, prim_path)
                
                # Calculate volume and other properties
                size = [aabb[i+3] - aabb[i] for i in range(3)]
                volume = size[0] * size[1] * size[2]
                center = [(aabb[i] + aabb[i+3])/2 for i in range(3)]
                
                components.append({
                    'path': prim_path,
                    'name': prim.GetName(),
                    'aabb': aabb,
                    'size': size,
                    'volume': volume,
                    'center': center
                })
                
                print(f"  Found component: {prim_path}")
                print(f"    Volume: {volume:.3f}, Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
                
            except Exception as e:
                print(f"Warning: Could not analyze component {prim.GetPath()}: {e}")
    
    return components


def classify_components(components):
    """Classify components into main building and decorative elements."""
    if len(components) == 0:
        return [], []
    elif len(components) == 1:
        # If only one component, try to analyze its internal structure
        return analyze_single_component(components[0])
    
    # Sort by volume (largest first)
    sorted_components = sorted(components, key=lambda x: x['volume'], reverse=True)
    
    # Strategy 1: Largest component is main building
    main_building = sorted_components[0]
    decorative_elements = []
    
    # Strategy 2: Classify based on position and size
    main_center_x = main_building['center'][0]
    
    for comp in sorted_components[1:]:
        # If component is significantly smaller and positioned differently
        if (comp['volume'] < main_building['volume'] * 0.3 and 
            abs(comp['center'][0] - main_center_x) > max(main_building['size']) * 0.3):
            decorative_elements.append(comp)
        else:
            # Merge with main building if similar size/position
            pass
    
    # If no decorative elements found, take the second largest as decorative
    if not decorative_elements and len(sorted_components) > 1:
        decorative_elements = [sorted_components[1]]
    
    return [main_building], decorative_elements


def analyze_single_component(component):
    """Analyze a single large component to try to separate building parts."""
    print(f"  Analyzing single component: {component['path']}")
    
    # For now, we'll simulate the separation based on spatial analysis
    # This would need to be enhanced based on actual mesh geometry
    main_parts = [component]
    decorative_parts = []
    
    # Create a virtual decorative element based on spatial assumption
    # Assuming the flower is on the left side of the building
    building_center = component['center']
    building_size = component['size']
    
    # Create virtual bounding boxes for sub-components
    # Main building (right and center part)
    main_bbox = component['aabb'].copy()
    main_bbox[0] = building_center[0] - building_size[0] * 0.3  # Reduce left boundary
    
    # Decorative element (left part - flower)
    decorative_bbox = component['aabb'].copy()
    decorative_bbox[3] = building_center[0] - building_size[0] * 0.3  # Set right boundary
    decorative_bbox[0] = component['aabb'][0]  # Keep original left boundary
    
    # Create virtual decorative component
    decorative_size = [decorative_bbox[i+3] - decorative_bbox[i] for i in range(3)]
    decorative_volume = decorative_size[0] * decorative_size[1] * decorative_size[2]
    decorative_center = [(decorative_bbox[i] + decorative_bbox[i+3])/2 for i in range(3)]
    
    if decorative_volume > 0:
        decorative_parts = [{
            'path': component['path'] + "_decorative",
            'name': component['name'] + "_flower",
            'aabb': decorative_bbox,
            'size': decorative_size,
            'volume': decorative_volume,
            'center': decorative_center,
            'virtual': True
        }]
        
        # Update main component to exclude decorative part
        main_parts[0] = {
            'path': component['path'] + "_main",
            'name': component['name'] + "_building",
            'aabb': main_bbox,
            'size': [main_bbox[i+3] - main_bbox[i] for i in range(3)],
            'volume': component['volume'] - decorative_volume,
            'center': [(main_bbox[i] + main_bbox[i+3])/2 for i in range(3)],
            'virtual': True,
            'original_path': component['path']
        }
    
    return main_parts, decorative_parts

def create_bounding_boxes(stage):
    """Create and display bounding boxes for model components."""
    print("\n" + "="*60)
    print("BOUNDING BOX ANALYSIS")
    print("="*60)
    
    # Create bounding box group
    bbox_group = prim_utils.define_prim("/World/BoundingBoxes", "Xform")
    
    # Analyze components
    components = analyze_model_components(stage)
    if not components:
        print("No mesh components found for bounding box analysis.")
        return
    
    print(f"Found {len(components)} mesh components")
    
    # 使用所有组件创建一个大的bounding box
    cache = create_bbox_cache()
    all_paths = [comp['path'] for comp in components]
    
    # 计算整体的组合bounding box
    if len(all_paths) == 1:
        combined_aabb = components[0]['aabb']
    else:
        combined_aabb = compute_combined_aabb(cache, all_paths)
    
    bbox_info = {
        'entire_model': {
            'paths': all_paths,
            'aabb': combined_aabb,
            'component_count': len(components)
        }
    }
    
    # 创建整个模型的OBB
    if args_cli.bbox_mode in ["obb", "both"]:
        try:
            centroid, axes, half_extent = compute_obb(cache, all_paths[0])
            corners = get_obb_corners(centroid, axes, half_extent)
            create_wireframe_cube(stage, corners, "EntireModel_OBB", (0.0, 0.8, 1.0))  # Cyan
            
            bbox_info['entire_model']['obb'] = {
                'centroid': centroid,
                'corners': corners
            }
            
            print(f"  Entire Model OBB created - Centroid: {centroid}")
            print(f"  OBB 8 Corners:")
            for i, corner in enumerate(corners):
                print(f"    Corner {i+1}: [{corner[0]:.3f}, {corner[1]:.3f}, {corner[2]:.3f}]")
                
        except Exception as e:
            print(f"  Warning: Could not create OBB: {e}")
    
    # 创建整个模型的AABB
    if args_cli.bbox_mode in ["aabb", "both"]:
        create_aabb_wireframe(stage, combined_aabb, "EntireModel", (0.0, 1.0, 0.0))  # Green
        size = [combined_aabb[i+3]-combined_aabb[i] for i in range(3)]
        print(f"  Entire Model AABB created - Size: {size}")
    
    print("="*60)
    return bbox_info

def print_model_info(stage, usd_path):
    """Print information about the USD model."""
    print("=" * 80)
    print(f"USD Model Information: {os.path.basename(usd_path)}")
    print("=" * 80)
    
    mesh_count = 0
    # Find the main model prim
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            mesh_count += 1
            print(f"Mesh Prim: {prim.GetPath()}")
            
            # Check for physics properties
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body = UsdPhysics.RigidBodyAPI(prim)
                print("  - Has Rigid Body API")
                
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision = UsdPhysics.CollisionAPI(prim)
                print("  - Has Collision API")
                
            if prim.HasAPI(UsdPhysics.MassAPI):
                mass_api = UsdPhysics.MassAPI(prim)
                mass = mass_api.GetMassAttr().Get()
                if mass:
                    print(f"  - Mass: {mass} kg")
                    
            # Get mesh info
            mesh = UsdGeom.Mesh(prim)
            if mesh:
                points = mesh.GetPointsAttr().Get()
                if points:
                    print(f"  - Vertices: {len(points)}")
                    
                faces = mesh.GetFaceVertexCountsAttr().Get()
                if faces:
                    print(f"  - Faces: {len(faces)}")
    
    print(f"\nTotal Mesh Prims: {mesh_count}")
    print("=" * 80)


def main():
    # check valid file path
    usd_path = args_cli.input
    if not os.path.isabs(usd_path):
        usd_path = os.path.abspath(usd_path)
    if not check_file_path(usd_path):
        raise ValueError(f"Invalid USD file path: {usd_path}")

    print("-" * 80)
    print(f"Loading USD file: {usd_path}")
    print("-" * 80)

    # Load the USD file first
    success = stage_utils.open_stage(usd_path)
    if not success:
        raise RuntimeError(f"Failed to load USD file: {usd_path}")

    # Get the stage
    stage = omni.usd.get_context().get_stage()

    # Setup physics if enabled
    if args_cli.physics:
        setup_physics(stage)

    # Add ground plane if not disabled
    if not args_cli.no_ground_plane:
        add_ground_plane(stage)

    # Setup camera
    setup_camera(stage, args_cli.camera_distance)

    # Show collision meshes if requested
    if args_cli.show_collision:
        show_collision_meshes(stage)

    # Print model information
    print_model_info(stage, usd_path)

    # Create bounding boxes if requested
    bbox_info = None
    if args_cli.show_bounding_boxes:
        bbox_info = create_bounding_boxes(stage)

    # Create simulation context (simplified)
    sim = None
    if args_cli.physics:
        try:
            sim = SimulationContext()
            sim.reset()
            sim.play()
        except Exception as e:
            print(f"Warning: Could not initialize physics simulation: {e}")
            print("Continuing without physics...")

    # Determine if there is a GUI to update
    carb_settings_iface = carb.settings.get_settings()
    local_gui = carb_settings_iface.get("/app/window/enabled")
    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

    # Run simulation loop (if not headless)
    if local_gui or livestream_gui:
        app = omni.kit.app.get_app_interface()
        print("=" * 80)
        print("USD Model Viewer Controls:")
        print("- Use mouse to orbit around the model")
        print("- Mouse wheel to zoom in/out")
        print("- Hold Shift + mouse to pan")
        print("- Press ESC or close window to exit")
        if args_cli.physics and sim:
            print("- Physics simulation is enabled")
        if args_cli.show_collision:
            print("- Collision meshes are shown in red")
        if args_cli.show_bounding_boxes:
            print("- Bounding boxes are displayed:")
            print("  • Main Building: Cyan (OBB) / Green (AABB)")
            print("  • Decorative Elements: Orange (OBB) / Magenta (AABB)")
            print(f"  • Mode: {args_cli.bbox_mode.upper()}")
        print("=" * 80)
        
        # Run simulation
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                if args_cli.physics and sim:
                    try:
                        sim.step()
                    except:
                        app.update()
                else:
                    app.update()
    else:
        print("Running in headless mode - no GUI available")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()