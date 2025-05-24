import numpy as np

rot = [0.941, 0.339, -0.003, -0.339, 0.941, -0.003, 0.002, 0.004, 1.000]
trans = [-1.199, 0.664, -0.372]

rot_mocap_to_map = np.array(rot).reshape(3, 3)
trans_mocap_in_map = np.array(trans).reshape(3, 1)

print("Rotation matrix (mocap to map):")
print(rot_mocap_to_map)
print("\nTranslation vector (mocap in map):")
print(trans_mocap_in_map)

anchor1_in_mocap = np.array([5.279, -1.058, 1.215]).reshape(3, 1)
anchor2_in_mocap = np.array([5.028, 1.625, 1.198]).reshape(3, 1) 
anchor3_in_mocap = np.array([-0.793, -2.566, 1.070]).reshape(3, 1)
anchor4_in_mocap = np.array([-3.517, 2.157, 0.785]).reshape(3, 1)

def transform_point(rot_mat, trans_vec, point):
    return rot_mat @ point + trans_vec  

anchor1_in_map = transform_point(rot_mocap_to_map, trans_mocap_in_map, anchor1_in_mocap)
anchor2_in_map = transform_point(rot_mocap_to_map, trans_mocap_in_map, anchor2_in_mocap)
anchor3_in_map = transform_point(rot_mocap_to_map, trans_mocap_in_map, anchor3_in_mocap)
anchor4_in_map = transform_point(rot_mocap_to_map, trans_mocap_in_map, anchor4_in_mocap)

print("\nConverted anchor points in map coordinate system:")
print(f"anchor1_in_map: {anchor1_in_map.flatten()}")
print(f"anchor2_in_map: {anchor2_in_map.flatten()}")
print(f"anchor3_in_map: {anchor3_in_map.flatten()}")
print(f"anchor4_in_map: {anchor4_in_map.flatten()}")

