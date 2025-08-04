from omni.isaac.orbit.app import AppLauncher

# 创建 AppLauncher，不传入任何 CLI 参数
app_launcher = AppLauncher()
simulation_app = app_launcher.app

import omni.kit.app
import contextlib
import carb

def main():
    print("Isaac Sim is running... Press ESC or close the window to exit.")
    app = omni.kit.app.get_app_interface()

    # 检查是否有 GUI
    carb_settings_iface = carb.settings.get_settings()
    local_gui = carb_settings_iface.get("/app/window/enabled")
    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

    # 持续运行窗口（非 headless）
    if local_gui or livestream_gui:
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                app.update()
    else:
        print("Running in headless mode - no GUI available")

if __name__ == "__main__":
    main()
    simulation_app.close()