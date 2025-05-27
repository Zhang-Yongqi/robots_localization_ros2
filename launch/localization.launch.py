##launch file
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share_dir = get_package_share_directory("robots_localization")
    config = os.path.join(pkg_share_dir, "config", "rse1r.yaml")
    pcd_path = os.path.join(pkg_share_dir, "PCD", "2024-11-24_19-24-09_scans.pcd")
    rviz_config = os.path.join(pkg_share_dir, "rviz", "localization.rviz")

    return LaunchDescription(
        [
            Node(
                package="robots_localization",
                executable="robots_localization_node",
                name="RobotsLocalizationNode",
                parameters=[config, {'pcd_path': pcd_path}],
                output="screen",
            ),
            # Node(
            #     package="rviz2",
            #     executable="rviz2",
            #     name="rviz2",
            #     arguments=["-d", rviz_config], 
            #     output="screen",
            # ),
        ]
    )
