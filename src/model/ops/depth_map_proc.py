import numpy as np 
import cv2 as cv 
import numba 
from numba import prange


@numba.njit()
def simple_depth_completion_inner(canvas, distance_record, start, step):
    INF = 1e8
    rows, cols = canvas.shape 
    current_pos = start
    # prev_depth = -1.0 
    prev_depth = 0
    prev_distance = INF
    step_length = np.sqrt(np.sum(step ** 2))
    while (
        current_pos[0] >= 0 and 
        current_pos[0] < rows and 
        current_pos[1] >= 0 and 
        current_pos[1] < cols 
    ):
        index = (current_pos[0], current_pos[1])
        # 修改需要符合NYU 和kitti 的数据情况，这里深度值无数据判断从-1 变成了0 
        # if canvas[index] == -1: 
        if canvas[index] == 0: 
            canvas[index] = prev_depth 
            distance_record[index] = prev_distance 
        else:  # != -1
            distance1 = distance_record[index] 
            distance2 = prev_distance
            if distance1 > distance2: 
                distance_record[index] = distance2
                canvas[index] = prev_depth 
            prev_depth = canvas[index]
            prev_distance = distance_record[index]
        prev_distance += step_length
        current_pos += step


@numba.njit(parallel=True)
def simple_depth_completion(depth):
    INF = 1e8
    rows, cols = depth.shape 
    canvas = np.copy(depth)
    distance_record = np.zeros((rows, cols), dtype=np.float32)
    for c in prange(cols):
        simple_depth_completion_inner(canvas, distance_record, np.array([0, c]), np.array([1, 0]))
        simple_depth_completion_inner(canvas, distance_record, np.array([rows - 1, c]), np.array([-1, 0]))
    for r in prange(rows):
        simple_depth_completion_inner(canvas, distance_record, np.array([r, 0]), np.array([0, 1]))
        simple_depth_completion_inner(canvas, distance_record, np.array([r, cols - 1]), np.array([0, -1]))
    return canvas, distance_record
            

def _simple_noise_filter_0(sparse_depth_map):
    sparse_depth_map = sparse_depth_map.copy()
    rows, cols = sparse_depth_map.shape 
    dense_depth, _ = simple_depth_completion(sparse_depth_map)
    for c in range(cols): 
        pre_depth = dense_depth[0, c]
        for r in range(1, rows):
            if dense_depth[r, c] <= pre_depth:
                pre_depth = dense_depth[r, c]
            else: 
                sparse_depth_map[r, c] = -1 
    return sparse_depth_map 
    
def _simple_noise_filter_2(sparse_depth_map, thresh=0.6):
    sparse_depth_map = sparse_depth_map.copy()
    rows, cols = sparse_depth_map.shape 
    dense_depth, _ = simple_depth_completion(sparse_depth_map)
    for c in range(cols): 
        pre_depth = dense_depth[0, c]
        for r in range(1, rows):
            if dense_depth[r, c] <= pre_depth + thresh:
                pre_depth = dense_depth[r, c]
            else: 
                sparse_depth_map[r, c] = -1 
    return sparse_depth_map 


@numba.njit(parallel=True)
def simple_noise_filter(sparse_depth_map, lambda_=1.5, max_age_ratio=60, max_depth=1e9):
    sparse_depth_map = sparse_depth_map.copy()
    rows, cols = sparse_depth_map.shape 
    dense_depth, _ = simple_depth_completion(sparse_depth_map)
    for c in prange(cols): 
        pre_depth = max_depth
        age = 0
        for r in range(0, rows):
            if dense_depth[r, c] <= pre_depth * lambda_:
                pre_depth = dense_depth[r, c]
                age = 0
            elif sparse_depth_map[r, c] >= 0: 
                sparse_depth_map[r, c] = -1
                age += 1 
                max_age = max(1, 1 / max(dense_depth[r, c], 1) * max_age_ratio)
                if age >= max_age: 
                    age = 0
                    pre_depth = max_depth 
    return sparse_depth_map 

def _simple_noise_filter_3(sparse_depth_map, size=3, thresh=1.5): 
    sparse_depth_map = sparse_depth_map.copy()
    rows, cols = sparse_depth_map.shape 
    dense_depth, _ = simple_depth_completion(sparse_depth_map)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, size))
    dense_depth_dilate = cv.erode(dense_depth, kernel, borderValue=-1)
    for r in range(rows): 
        for c in range(cols): 
            if sparse_depth_map[r, c] >= 0 and sparse_depth_map[r, c] > dense_depth_dilate[r, c] + thresh: 
                sparse_depth_map[r, c] = -1 
    return sparse_depth_map

