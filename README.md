# RM2023 Sentry Localization

RoboMaster2023 哨兵定位代码，基于FAST LIO 2进行开发

## 环境配置

- Sophs

    ```bash
    git clone https://github.com/strasdat/Sophus.git
    cd Sophus/
    mkdir build
    cd build
    cmake ..
    make
    sudo make install
    ```

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
- Livox MID360环境配置\
先下载`Livox-SDK2`:

    ```bash
    git clone https://github.com/Livox-SDK/Livox-SDK2.git
    cd ./Livox-SDK2/
    mkdir build
    cd build
    cmake .. && make -j
    sudo make install
    ```

    然后在工作空间下下载`livox_ros_driver2`

    ```bash
    source /opt/ros/noetic/setup.sh
    git clone https://github.com/Livox-SDK/livox_ros_driver2.git
    cd livox_ros_driver2
    ./build.sh ROS1
    ```
    
    ```bash
    sudo apt install ccache
    ```

## Tips

连不上雷达优先查ip的问题，电脑不要挂代理，跟着livox_ros_driver的readme操作就行。

Livox的imu数据重力单位是g，注意代码里的归一化用的norm是多少。

livox_ros_driver2和livox_ros_driver的msg是一样的，一些依赖livox_ros_driver的代码用mid360时也不需要改成livox_ros_driver2。ws里包含livox_ros_driver，另外用livox_ros_driver2去打开mid360即可。

livox_ros_driver的config里改外参这一功能主要是提供给多雷达方案的，改变的是雷达数据的坐标系。

mid360并非机械式，没有line的说法，但是scan_line一般设置成4

IMU在点云坐标系下的位置为[0.011, 0.0234, -0.044] ，填写外参时注意代码中的基坐标系是哪个

header中的time是该帧雷达的最早时间，point的time是相对于其header的time的offset，unit: ms

## Usage

- MID360

    ```bash
    roslaunch livox_ros_driver2 msg_MID360.launch
    roslaunch robots_localization localization_mid360.launch
    ```

## 可能出现的问题

1. 编译代码时如果内存空间不足可能会导致编译失败，如果出现编译卡死然后崩溃的情况可以扩充swap空间：
    参考[扩充Swap空间](https://blog.csdn.net/epic_Lin/article/details/122369604?spm=1001.2014.3001.5501)
    如果不想设置为自动加载并清理创建的空间则直接把对应文件删除即可
