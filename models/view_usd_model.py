# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to view USD models in Isaac Sim.

This script uses Isaac Sim to load and visualize USD files with their properties
including collision meshes, mass properties, and materials.

positional arguments:
  input               The path to the input USD file to view.

optional arguments:
  -h, --help          Show this help message and exit
  --physics           Enable physics simulation (default: False)
  --gravity           Set gravity value (default: -9.81)
  --show-collision    Show collision meshes (default: False)
  --camera-distance   Set camera distance from model (default: 5.0)
  --ground-plane      Add ground plane (default: True)
"""

"""Launch Isaac Sim Simulator first."""
"""python view_usd_model.py "/home/hu/CaricRL/CoverageRL/isaac-training/training/models/mbs/mbs.usd"""
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

import carb
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.prims as prim_utils
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


def print_model_info(stage, usd_path):
    """Print information about the USD model."""
    print("=" * 80)
    print(f"USD Model Information: {os.path.basename(usd_path)}")
    print("=" * 80)
    
    # Find the main model prim
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
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

