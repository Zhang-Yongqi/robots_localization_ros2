# Localization

定位代码，基于FAST LIO 2进行开发

## TODO

1. 建图加入GTSAM等回环检测
2. 增加关键帧机制，帮助重定位

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

- ccache
  
    ```bash
    sudo apt install ccache
    ```
- ceres
  
    ```bash
    sudo apt install libceres-dev
    ```

## Tips

连不上雷达优先查ip的问题，电脑不要挂代理。

Livox的imu数据重力单位是g，注意代码里的归一化用的norm是多少。

IMU在点云坐标系下的位置为[0.011, 0.0234, -0.044] ，填写外参时注意代码中的基坐标系是哪个

header中的time是该帧雷达的最早时间，point的time是相对于其header的time的offset，unit: ms
