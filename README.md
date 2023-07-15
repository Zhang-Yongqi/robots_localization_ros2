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
MID360依赖`livox_ros_driver2`这个Package，注意不要安装错了. 但是这个包依赖`livox_ros_driver`这个Package(Horizon对应驱动)，因此其实也是需要安装的。最好的解决措施是在包的msg里面自定义上述两个驱动中的msg消息并生成，然后将代码中所有涉及到`livox_ros_driver::CustomMsg`的地方全部改成自定义的msg，例如`robots_localization::CustomMsg`.只要定义的消息类型一直，ros就能完成订阅的解析，不需要依赖额外的package(2023-03-20已修复)\
先下载`Livox-SDK2`:
    ```bash
    $ git clone https://github.com/Livox-SDK/Livox-SDK2.git
    $ cd ./Livox-SDK2/
    $ mkdir build
    $ cd build
    $ cmake .. && make -j
    $ sudo make install
    ```
    然后在工作空间下下载`livox_ros_driver2`
    ```bash
    source /opt/ros/noetic/setup.sh
    git clone https://github.com/Livox-SDK/livox_ros_driver2.git
    cd livox_ros_driver2
    ./build.sh ROS1
    ```
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
