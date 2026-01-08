# Localization

定位代码，基于FAST LIO 2进行开发

## 主要参数说明

- `mapping_en`：建图与定位模式切换参数。
    - `false`：定位模式，仅进行定位，不生成地图。
    - `true`：建图模式，同时进行建图与定位，会在PCD文件夹中生成两个pcd文件，建议后续定位时使用`*_ikdtree.pcd`。
- `imu2robot_T`、`imu2robot_R`：IMU系到机器人本体的外参，分别为平移和旋转矩阵，需根据实际安装情况填写。

### 使用建议
1. **首次使用/未建图时**：请先将`mapping_en`设为`true`，完成建图。
2. **定位时**：将`mapping_en`设为`false`，使用已生成的`*_ikdtree.pcd`进行定位。

> 建议流程：先建图（mapping_en: true），再定位（mapping_en: false）。

## TODO

1. 建图加入GTSAM等回环检测
2. 增加关键帧机制，帮助重定位

## 环境配置

- Sophus

    ```bash
    git clone https://github.com/strasdat/Sophus.git
    cd Sophus/
    git checkout 1.22.10
    mkdir build
    cd build
    cmake ..
    make
    sudo make install
    ```

    尽量选择较老版本，不需要升级Cmake版本。

- fmt

    ```bash

    git clone https://github.com/fmtlib/fmt
    cd fmt
    mkdir build
    cd build
    cmake ..
    make
    sudo make install
    ```

    注意必须在CMakeLists中添加``add_compile_options(-fPIC)`` ！！

- ccache
  
    ```bash
    sudo apt install ccache
    ```

- ceres
  
    ```bash
    sudo apt install libceres-dev
    ```

## Tips

如果rviz中没有参考点云，请进行pointcloud显示设置
RViz配置要求
在RViz的PointCloud2显示设置中,需要确保:

- Durability Policy: Transient Local (不是 Volatile)
- Reliability: Reliable (推荐)
- History: Keep Last

确保有PCD文件夹，否则会编译失败

连不上雷达优先查ip的问题，电脑不要挂代理。

Livox的imu数据重力单位是g，注意代码里的归一化用的norm是多少。

mid360的IMU在点云坐标系下的位置为[0.011, 0.0234, -0.044] ，填写外参时注意代码中的基坐标系是哪个

header中的time是该帧雷达的最早时间，point的time是相对于其header的time的offset，unit: ms。

推荐编译方式: colcon build --symlink-install --cmake-args -G Ninja
