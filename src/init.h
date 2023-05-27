#include <ros/ros.h>
// 由于要用定时器回调函数，这些Publisher定义在int main中无法被回调函数访问，在这里声明
ros::Publisher pubLaserCloudFull;
ros::Publisher pubLaserCloudFull_body;
ros::Publisher pubLaserCloudFull_world;
ros::Publisher pubLaserCloudMap;
ros::Publisher pubOdomAftMapped;
ros::Publisher pubPath;
ros::Publisher pubVelo;
ros::Time start_time;
ros::Time end_time;
ros::Duration duration_time;
ros::Time start2;
ros::Time end2;
ros::Duration duration2;
ros::Time start3;
ros::Time end3;
ros::Duration duration3;

void process_lidar();