#!/usr/bin/env python3
import rosbag
import bisect
import os
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin

# 自定义固定斜率回归器
class FixedSlopeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, slope=1.0):
        self.slope = slope
        self.intercept_ = 0.0
    
    def fit(self, X, y):
        residuals = y - self.slope * X.ravel()
        self.intercept_ = np.mean(residuals)
        return self
    
    def predict(self, X):
        return self.slope * X.ravel() + self.intercept_

BAG_PATH = '/home/xwl/2025-04-17-16-44.bag'                          # ROS bag文件路径
MOCAP_TOPIC = '/mavros/vision_pose/pose02'                 # 动捕数据话题
UWB_TOPIC = '/linktrack2/nlink_linktrack_nodeframe2'       # UWB数据话题
UWB_ANCHOR_IDS = [1, 2, 3, 4]                              # UWB基站ID列表
ANCHOR_POSITIONS = {                                       # 动捕系下基站坐标 (x, y, z)
    1: [5.279, -1.058, 1.215],
    2: [5.028, 1.625, 1.198],
    3: [-0.793, -2.566, 1.070],
    4: [-3.517, 2.157, 0.785]
}
offset = [0.1, 0.0, 0.15]

mocap_data = []     # (timestamp, [x, y, z])
uwb_data = []        # (timestamp, {anchor_id: distance})

print("正在读取ROS bag文件...")
with rosbag.Bag(BAG_PATH, 'r') as bag:
    for topic, msg, _ in bag.read_messages(topics=[MOCAP_TOPIC]):
        if topic == MOCAP_TOPIC:
            mocap_time = msg.header.stamp.to_sec()
            position = [
                msg.pose.position.x - offset[0],
                msg.pose.position.y - offset[1],
                msg.pose.position.z - offset[2]
            ]
            mocap_data.append((mocap_time, position))
    
    for topic, msg, _ in bag.read_messages(topics=[UWB_TOPIC]):
        if topic == UWB_TOPIC:
            uwb_time = msg.timestamp.to_sec()
            anchor_dist = {}
            for node in msg.nodes:
                node_id = int(node.id)
                if node_id in UWB_ANCHOR_IDS:
                    anchor_dist[node_id] = float(node.dis)
            uwb_data.append((uwb_time, anchor_dist))

mocap_data.sort(key=lambda x: x[0])
mocap_timestamps = [t for t, _ in mocap_data]
duration = mocap_timestamps[-1] - mocap_timestamps[0]

data_pairs = {aid: {'mocap': [], 'uwb': []} for aid in UWB_ANCHOR_IDS}

print("正在处理数据对齐...")
for uwb_time, uwb_measurements in uwb_data:
    idx = bisect.bisect_left(mocap_timestamps, uwb_time)
    candidates = []
    if idx > 0: candidates.append(idx-1)
    if idx < len(mocap_timestamps): candidates.append(idx)
    
    min_diff, best_idx = float('inf'), -1
    for ci in candidates:
        diff = abs(mocap_timestamps[ci] - uwb_time)
        if diff < min_diff:
            min_diff, best_idx = diff, ci
    
    if best_idx == -1 or min_diff > 0.02:
        continue
    
    mocap_pos = mocap_data[best_idx][1]
    
    for aid, meas_dist in uwb_measurements.items():
        if aid not in ANCHOR_POSITIONS:
            continue
        
        anchor_pos = ANCHOR_POSITIONS[aid]
        dx = mocap_pos[0] - anchor_pos[0]
        dy = mocap_pos[1] - anchor_pos[1]
        dz = mocap_pos[2] - anchor_pos[2]
        mocap_dist = np.sqrt(dx**2 + dy**2 + dz**2)
        
        data_pairs[aid]['mocap'].append(mocap_dist)
        data_pairs[aid]['uwb'].append(meas_dist)

print("正在生成图表并进行RANSAC拟合...")
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
axs = axs.ravel()

bag_name = os.path.basename(BAG_PATH).replace('.bag', '')

for idx, anchor_id in enumerate(UWB_ANCHOR_IDS):
    ax = axs[idx]
    mocap = np.array(data_pairs[anchor_id]['mocap']).reshape(-1, 1)  
    uwb = np.array(data_pairs[anchor_id]['uwb'])
    
    if len(mocap) < 2 or len(uwb) < 2:
        print(f"锚点 {anchor_id} 数据不足，跳过拟合")
        continue
    
    try:
        res_threshold = 0.4
        fix_scale_1 = True
        if fix_scale_1:
            # 使用自定义估计器，固定斜率为1
            base_estimator = FixedSlopeRegressor(slope=1.0)
            ransac = RANSACRegressor(base_estimator=base_estimator,
                                    min_samples=0.8,                  # 至少80%样本作为内点
                                    residual_threshold=res_threshold, # 残差阈值(m)
                                    max_trials=1000)                  # 最大迭代次数
            ransac.fit(mocap, uwb)
            inlier_mask = ransac.inlier_mask_
            
            scale = ransac.estimator_.slope
        else:
            ransac = RANSACRegressor(min_samples=0.8,                 # 至少80%样本作为内点
                                    residual_threshold=res_threshold, # 残差阈值(m)
                                    max_trials=1000)                  # 最大迭代次数
            ransac.fit(mocap, uwb)
            inlier_mask = ransac.inlier_mask_
            
            scale = ransac.estimator_.coef_[0]

        bias = ransac.estimator_.intercept_
        
        x_range = np.linspace(min(mocap), max(mocap), 100)
        y_pred = ransac.predict(x_range)
        
        ax.scatter(mocap[inlier_mask], uwb[inlier_mask], 
                  s=5, c='b', alpha=0.6, label='inlier')
        ax.scatter(mocap[~inlier_mask], uwb[~inlier_mask],
                  s=5, c='r', alpha=0.6, label='outlier')
        ax.plot(x_range, y_pred, 'g-', lw=2, 
               label=f'ransac fit: y={scale:.3f}x + {bias:.3f}')
        
    except Exception as e:
        print(f"锚点 {anchor_id} 拟合失败: {str(e)}")
        scale, bias = np.nan, np.nan
        ax.scatter(mocap, uwb, alpha=0.6)
    
    max_val = max(mocap.max() if len(mocap) else 0, 
                 uwb.max() if len(uwb) else 10)
    ax.plot([0, max_val], [0, max_val], 'k--', label='Ideal')
    
    ax.set_xlabel('Mocap Distance (m)')
    ax.set_ylabel('UWB Distance (m)')
    ax.legend(loc='upper left')
    ax.set_title(f'anchor {anchor_id}\n'
                f'fit param: scale={scale:.3f} bias={bias:.3f}, residual_threshold={res_threshold:.2f}m')
    ax.grid(True)
    ax.set_xlim(0, max_val*1.1)
    ax.set_ylim(0, max_val*1.1)

plt.suptitle(f'Bag: {bag_name}, duration={duration:.2f}s')

plt.tight_layout()
plt.show()

print("处理完成！")