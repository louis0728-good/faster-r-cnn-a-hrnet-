import torch
import numpy as np
import ipdb
import glob
import os
import io
import math
import random
import json
import pickle
import math
from torch.utils.data import Dataset, DataLoader
from lib.utils.utils_data import crop_scale

"""
def halpe2h36m(x):
    '''
        Input: x (T x V x C)  
       //Halpe 26 body keypoints
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "Rknee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
    {17,  "Head"},
    {18,  "Neck"},
    {19,  "Hip"},
    {20, "LBigToe"},
    {21, "RBigToe"},
    {22, "LSmallToe"},
    {23, "RSmallToe"},
    {24, "LHeel"},
    {25, "RHeel"},
    '''
    T, V, C = x.shape
    y = np.zeros([T,17,C])
    y[:,0,:] = x[:,19,:]
    y[:,1,:] = x[:,12,:]
    y[:,2,:] = x[:,14,:]
    y[:,3,:] = x[:,16,:]
    y[:,4,:] = x[:,11,:]
    y[:,5,:] = x[:,13,:]
    y[:,6,:] = x[:,15,:]
    y[:,7,:] = (x[:,18,:] + x[:,19,:]) * 0.5
    y[:,8,:] = x[:,18,:]
    y[:,9,:] = x[:,0,:]
    y[:,10,:] = x[:,17,:]
    y[:,11,:] = x[:,5,:]
    y[:,12,:] = x[:,7,:]
    y[:,13,:] = x[:,9,:]
    y[:,14,:] = x[:,6,:]
    y[:,15,:] = x[:,8,:]
    y[:,16,:] = x[:,10,:]
    return y
"""

def coco2h36m(x):
    """
    將 COCO 17 關鍵點轉換為 H36M 17 關鍵點
    Input: x (T x V x C)  
    格式會長得像，T = 時間(幀數), V=關鍵點數(17), C=座標維度(x,y,v)
    
    COCO_KEYPOINTS_17_SIMPLE = {
        0: '鼻子: nose',
        1: '左眼: left_eye',
        2: '右眼: right_eye',
        3: '左耳: left_ear',
        4: '右耳: right_ear',
        5: '左肩: left_shoulder',
        6: '右肩: right_shoulder',
        7: '左手肘: left_elbow',
        8: '右手肘: right_elbow',
        9: '左手腕: left_wrist',
        10: '右手腕: right_wrist',
        11: '左髖 (左臀): left_hip',
        12: '右髖 (右臀): right_hip',
        13: '左膝: left_knee',
        14: '右膝: right_knee',
        15: '左腳踝: left_ankle',
        16: '右腳踝: right_ankle'
    }
    
    H36M_KEYPOINTS_17_SIMPLE = {
        0: '髖關節 (中心): Hip',
        1: '右髖 (右臀): RHip',
        2: '右膝: RKnee',
        3: '右腳踝: RAnkle',
        4: '左髖 (左臀): LHip',
        5: '左膝: LKnee',
        6: '左腳踝: LAnkle',
        7: '脊椎 (中心): Spine',
        8: '胸腔 (中心): Thorax',
        9: '鼻子: Nose',
        10: '頭部: Head',
        11: '左肩: LShoulder',
        12: '左手肘: LElbow',
        13: '左手腕: LWrist',
        14: '右肩: RShoulder',
        15: '右手肘: RElbow',
        16: '右手腕: RWrist'
    }
    """

    T, V, C = x.shape # 取得 NumPy 陣列 x 的維度
    y = np.zeros([T, 17, C])
    
    # 直接對應的關鍵點
    y[:, 9, :] = x[:, 0, :]    # Nose
    y[:, 11, :] = x[:, 5, :]   # Left Shoulder
    y[:, 12, :] = x[:, 7, :]   # Left Elbow
    y[:, 13, :] = x[:, 9, :]   # Left Wrist
    y[:, 14, :] = x[:, 6, :]   # Right Shoulder
    y[:, 15, :] = x[:, 8, :]   # Right Elbow
    y[:, 16, :] = x[:, 10, :]  # Right Wrist
    y[:, 1, :] = x[:, 12, :]   # Right Hip
    y[:, 2, :] = x[:, 14, :]   # Right Knee
    y[:, 3, :] = x[:, 16, :]   # Right Ankle
    y[:, 4, :] = x[:, 11, :]   # Left Hip
    y[:, 5, :] = x[:, 13, :]   # Left Knee
    y[:, 6, :] = x[:, 15, :]   # Left Ankle
    
    # 需要計算的關鍵點
    # Hip(髖中心) = 左右髖的中點
    y[:, 0, :] = (x[:, 11, :] + x[:, 12, :]) * 0.5
    
    # thorax(胸部) = 左右肩的中點
    y[:, 8, :] = (x[:, 5, :] + x[:, 6, :]) * 0.5
    
    # Spine(脊椎) = Hip 和 Thorax 的中點
    y[:, 7, :] = (y[:, 0, :] + y[:, 8, :]) * 0.5
    
    # Head(頭頂) = 鼻子正上方，距離約為肩寬的 0.2 倍
    shoulder_width = np.sqrt(
        (x[:, 5, 0] - x[:, 6, 0])**2 + 
        (x[:, 5, 1] - x[:, 6, 1])**2
    )

    y[:, 10, :] = x[:, 0, :].copy()
    y[:, 10, 1] = x[:, 0, 1] - shoulder_width * 0.2  # y 減少 = 往上
    
    return y


def read_input(json_path, vid_size, scale_range, focus):
    # 檢查 json_path 是資料夾還是檔案
    if os.path.isdir(json_path):
        # 如果是資料夾，讀取所有 json 檔案
        json_files = sorted(glob.glob(os.path.join(json_path, "*.json")))
        # glob.glob(...)  會找出所有符合的檔案
        # sorted(...) 會按檔名排序

        if len(json_files) == 0:
            raise ValueError(f"在 {json_path} 中找不到任何 json 檔案")
    else:
        # 如果是單一檔案（保持向後兼容）
        json_files = [json_path]


    # 用來存放所有幀的關鍵點資料
    kpts_all = []
    frame_data = []  # 用來存放 (幀數, 關鍵點) 的配對
    
    # 逐個讀取 json 檔案
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 提取幀數（用於排序）
        frame_num = data.get('frame', 0)
        keypoints_list = data.get('keypoints', [])
        
        # 檢查是否有 17 個關鍵點
        if len(keypoints_list) != 17:
            print(f"警告：{json_file} 只有 {len(keypoints_list)} 個關鍵點，跳過此幀，怪怪的。\n")
            continue
        
        # 轉換成 numpy array (17, 3)
        temp = np.zeros((17, 3))
        for k in keypoints_list:
            idx = k['id']  # 關鍵點的 id (0-16)
            temp[idx, 0] = k['x']
            temp[idx, 1] = k['y']
            temp[idx, 2] = k['v']
        
        # 存放 (幀數, 關鍵點)
        frame_data.append((frame_num, temp))
    
    # 按照幀數排序
    frame_data.sort(key=lambda x: x[0])

    # 填補缺失的幀（我會讓他暫停在原地，因為我覺得變空白很怪）
    if len(frame_data) > 0:

        # 找出第一幀和最後一幀的幀數
        first_frame = frame_data[0][0]
        last_frame = frame_data[-1][0]
        
        # 建立完整的幀數範圍 [first_frame, first_frame+1, ..., last_frame]
        expected_frames = set(range(first_frame, last_frame + 1))
        
        # 找出實際存在的幀數
        existing_frames = {frame_num for frame_num, _ in frame_data}
        # 例如：{1, 2, 3, 5, 6, 7, ...}（第 4 幀缺失）
        
        # 找出缺失的幀數
        missing_frames = expected_frames - existing_frames
        
        if len(missing_frames) > 0:
            print(f"警告：發現 {len(missing_frames)} 個缺失的幀，將用前一幀填補")


        # 建立幀數到關鍵點的映射（方便查詢）
        frame_dict = {frame_num: kpts for frame_num, kpts in frame_data}
        
        # 填補缺失的幀
        for missing_frame in sorted(missing_frames):
            # 如果缺失 {4, 7, 9}，按 4 → 7 → 9 的順序處理
            # 找前一幀（前一個存在的幀）
            prev_frame = missing_frame - 1
            
            # 如果前一幀存在，就複製它的關鍵點
            if prev_frame in frame_dict:
                frame_dict[missing_frame] = frame_dict[prev_frame].copy()
                print(f"  幀 {missing_frame} 缺失，使用幀 {prev_frame} 的姿態")

            else:
                # 如果前一幀也不存在，往前繼續找
                for i in range(missing_frame - 2, first_frame - 1, -1):
                    if i in frame_dict:
                        frame_dict[missing_frame] = frame_dict[i].copy()
                        print(f"  幀 {missing_frame} 缺失，使用幀 {i} 的姿態")
                        break
        
        # 重建 frame_data（包含填補後的幀）
        frame_data = [(frame_num, kpts) for frame_num, kpts in frame_dict.items()]
        frame_data.sort(key=lambda x: x[0])  # 重新排序

    
    # 只取關鍵點資料（去掉幀數）
    kpts_all = [kpts for _, kpts in frame_data]
    
    # 轉換成 numpy array: (T, 17, 3) (幾幀, 關鍵點數量, {x, y, v})
    kpts_all = np.array(kpts_all)
    # 所以現在裡面會有(000000001, 17 個點 ; 00000000002......)

    kpts_all = coco2h36m(kpts_all)

    motion = kpts_all
    if vid_size:
        w, h = vid_size
        scale = min(w,h) / 2.0
        kpts_all[:,:,:2] = kpts_all[:,:,:2] - np.array([w, h]) / 2.0
        kpts_all[:,:,:2] = kpts_all[:,:,:2] / scale
        motion = kpts_all
    if scale_range:
        motion = crop_scale(kpts_all, scale_range) 
    return motion.astype(np.float32)

class WildDetDataset(Dataset):
    def __init__(self, json_path, clip_len=243, vid_size=None, scale_range=None, focus=None):
        self.json_path = json_path
        self.clip_len = clip_len
        self.vid_all = read_input(json_path, vid_size, scale_range, focus)
        
    def __len__(self):
        'Denotes the total number of samples'
        if len(self.vid_all) == 0:
            return 0
        return math.ceil(len(self.vid_all) / self.clip_len)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        st = index*self.clip_len
        end = min((index+1)*self.clip_len, len(self.vid_all))

        data = self.vid_all[st:end]
        seq_len = data.shape[0]
        if seq_len < self.clip_len:
            # 複製最後一幀來進行填充
            padding = np.repeat(data[-1:], self.clip_len - seq_len, axis=0)
            # 拼接到原數據後面，確保總長度為 clip_len
            data = np.concatenate((data, padding), axis=0)
            
        return data