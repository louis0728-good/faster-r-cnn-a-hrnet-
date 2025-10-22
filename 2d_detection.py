from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import glob
import json
import numpy as np

import sys
#sys.path.append("../lib")
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

# 取得當前檔案的目錄，然後找到專案根目錄和 lib 目錄
current_dir = osp.dirname(osp.abspath(__file__))  # demo/

project_root = osp.dirname(current_dir)            # deep-high-resolution-net.pytorch/
# 往上一層到專案根目錄
lib_path = osp.join(project_root, 'lib')           # deep-high-resolution-net.pytorch/lib/
# 找到 lib/
sys.path.insert(0, project_root)
sys.path.insert(0, lib_path)

import time

import models
from config import cfg
from config import update_config
from core.inference import get_final_preds
from utils.transforms import get_affine_transform

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    print(" 嗨嗨，你現在是用 GPU peko !")
else:
    print("你用到 CPU 了 ! ")

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_SKELETON = [
    (0, 1), (0, 2),  # 鼻子到眼睛
    (1, 3), (2, 4),  # 眼睛到耳朵  
    (5, 6),          # 肩膀之間
    (5, 7), (7, 9),  # 左手臂
    (6, 8), (8, 10), # 右手臂
    (5, 11), (6, 12),# 肩膀到髖部
    (11, 12),        # 髖部之間
    (11, 13), (13, 15), # 左腿
    (12, 14), (14, 16)  # 右腿
]

def get_pose_estimation_prediction(pose_model, image, centers, scales, transform):
    rotation = 0

    # pose estimation transformation
    model_inputs = []
    for center, scale in zip(centers, scales):
        trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
        # Crop smaller image of people
        model_input = cv2.warpAffine(
            image,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

        # hwc -> 1chw
        model_input = transform(model_input)#.unsqueeze(0)
        model_inputs.append(model_input)

    # n * 1chw -> nchw
    model_inputs = torch.stack(model_inputs)

    # compute output heatmap
    output = pose_model(model_inputs.to(CTX))
    coords, confidence = get_final_preds(
        cfg,
        output.cpu().detach().numpy(),
        np.asarray(centers),
        np.asarray(scales))

    return coords, confidence

def draw_skeleton_and_keypoints(frame, keypoints, confidence):
    """ 畫骨架和關鍵點 """
    h, w = frame.shape[:2]
    
    # 先畫連線
    for c in COCO_SKELETON:
        c_id1, c_id2 = c
        if c_id1 < len(keypoints) and c_id2 < len(keypoints):
            x1, y1 = int(keypoints[c_id1][0]), int(keypoints[c_id1][1])
            x2, y2 = int(keypoints[c_id2][0]), int(keypoints[c_id2][1])
            
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
    
    # 再畫關鍵點
    for idx, (keypoint, conf) in enumerate(zip(keypoints, confidence)):
        x, y = int(keypoint[0]), int(keypoint[1])
        visibility = float(conf[0])
        
        if not (0 <= x < w and 0 <= y < h):
            continue
        
        # 根據信心度決定顏色
        if visibility > 0.8:
            color = (0, 255, 0)  # 綠色
        elif visibility >= 0.5:
            color = (0, 165, 255)  # 深橘色
        else:
            color = (0, 0, 255)  # 紅色
        
        cv2.circle(frame, (x, y), 4, color, -1)
        cv2.putText(frame, f"{visibility:.2f}", (x + 5, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
def load_bbox_from_json(json_path):
    """ 從 json 檔案載入 bbox 資訊 """
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # bbox 格式: [x, y, w, h] -> 轉換為 [(x1, y1), (x2, y2)]
    x, y, w, h = data['bbox']
    bbox = [(x, y), (x + w, y + h)]
    return bbox, data['width'], data['height']


def save_keypoints_json(json_path, video_name, frame_number, fps, keypoints, confidence):
    """ 儲存關鍵點資料到 json """
    landmarks_data = {
        "video_name": video_name,
        "frame": frame_number,
        "keypoints": []
    }
    
    for idx in range(len(keypoints)):
        keypoint_name = COCO_KEYPOINT_INDEXES.get(idx, f"unknown_{idx}")
        x, y = keypoints[idx]
        v = float(confidence[idx][0])
        
        landmarks_data["keypoints"].append({
            "id": idx,
            "name": keypoint_name,
            "x": int(x),
            "y": int(y),
            "v": v
        })
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(landmarks_data, f, indent=2, ensure_ascii=False)


def save_empty_json(json_path, video_name, frame_number, fps):
    """ 儲存空的 json（假設拉 目前幀沒有對應 id 的數據，希望不會有）"""
    empty_data = {
        "video_name": video_name,
        "frame_number": frame_number,
        "timestamp": frame_number / fps,
        "keypoints": []
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(empty_data, f, indent=2, ensure_ascii=False)

def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    #parser.add_argument('--videoFile', type=str, required=True)
    parser.add_argument('--input_dir', type=str,
                       default=r'D:\rcnn\PyTorch-Object-Detection-Faster-RCNN-Tutorial\test_videos')
                        # 這路徑是死的，你們要改可以自己改
    # 因為我有自己的 faster r-cnn 所以我們這邊多加一個設定，bbox 的來源路徑
    parser.add_argument('--bbox_base', type=str,
                       default=r'D:\rcnn\PyTorch-Object-Detection-Faster-RCNN-Tutorial\output_videos')
    
    #parser.add_argument('--outputDir', type=str, default='/output/')
    parser.add_argument('--output_dir', type=str, 
                       default=r'D:\rcnn\PyTorch-Object-Detection-Faster-RCNN-Tutorial\output_videos\2d_detections')
    
    #parser.add_argument('--inferenceFps', type=int, default=10)
    #parser.add_argument('--writeBoxFrames', action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)
    
    


    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    # 我仍然保留原作者的定義，只是我在 else的地方改為我下載的 hrnet 權重
    if cfg.TEST.MODEL_FILE:
        print('=> 原作者流程匯入權重 {}'.format(cfg.TEST.MODEL_FILE))
        # pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
        # CTX 是我定義的 CUDA 使用，我要用 GPU 而不是 CPU
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location=CTX), strict=False)

    else:
        weights_path = r'D:\rcnn\deep-high-resolution-net.pytorch\demo\pose_hrnet_w48_384x288.pth'
        print(f'=> 已載入自定義的權重喵 ~ {weights_path}')
        # pose_model.load_state_dict(torch.load(weights_path), strict=False)
        pose_model.load_state_dict(torch.load(weights_path, map_location=CTX), strict=False)


    pose_model.to(CTX)
    pose_model.eval()

    # Loading an video
    """ 
    # 這裡是原作者的定義，我先改調好了，錯了再說
    vidcap = cv2.VideoCapture(args.videoFile)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if fps < args.inferenceFps:
        print('desired inference fps is '+str(args.inferenceFps)+' but video fps is '+str(fps))
        exit()
    skip_frame_cnt = round(fps / args.inferenceFps)
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outcap = cv2.VideoWriter('{}/{}_pose.avi'.format(args.outputDir, os.path.splitext(os.path.basename(args.videoFile))[0]),
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int(skip_frame_cnt), (frame_width, frame_height))
    """
    
    video_files = glob.glob(os.path.join(args.input_dir, '*.mp4'))

    if not video_files:
        print(f'欸欸: 在 {args.input_dir} 找不到任何 mp4 影片')
        return

    print(f'找到 {len(video_files)} 個影片等待處理\n')

    for v in video_files:
        video_name = os.path.splitext(os.path.basename(v))[0]
        print(f'正在處理: {video_name}.mp4')
        
        """ 處理輸出路徑 """
        video_output_dir = os.path.join(args.output_dir, video_name)
        img_dir = os.path.join(video_output_dir, "img")
        id1_dir = os.path.join(video_output_dir, "1")
        id2_dir = os.path.join(video_output_dir, "2")
        
        os.makedirs(video_output_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(id1_dir, exist_ok=True)
        os.makedirs(id2_dir, exist_ok=True)

        output_video_path = os.path.join(args.output_dir, f'{video_name}.mp4')
        
        cap = cv2.VideoCapture(v)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 輸出影片設定（並排）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, 
                            (frame_w * 2, frame_h))
        
        # BBox JSON 資料夾
        bbox_id1_dir = os.path.join(args.bbox_base, video_name, '1')
        bbox_id2_dir = os.path.join(args.bbox_base, video_name, '2')


        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            count += 1

            if not ret:
                break

            origi_frame = frame.copy() # 原始影片備份

            # 建立影片背景，我就直接把 mediapipe 那部分的照搬過來了，就是背景會有淺色系人物，方便你們看
            alpha = 0.5 # 透明度，數值越小越淡 (0.0~1.0) 
            skeleton_frame = cv2.addWeighted(frame, alpha, 
                                            np.full_like(frame, 255), 1 - alpha, 0)
            
            """ 若不要，想要骨架用白色單純背景，和原始圖片分離。"""
            #skeleton_frame = np.full((h, w, 3), 255, dtype=np.uint8) # 右邊是骨架影片
            
            # 讀取 bbox json
            json_filename = f'frame_{count:012d}.json'
            id1_json_path = os.path.join(bbox_id1_dir, json_filename)
            id2_json_path = os.path.join(bbox_id2_dir, json_filename)
            
            id1_data = load_bbox_from_json(id1_json_path)
            id2_data = load_bbox_from_json(id2_json_path)
            
            # 準備圖像
            if cfg.DATASET.COLOR_RGB:
                image_pose = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                image_pose = frame.copy()
            
            # 收集所有 bbox
            all_boxes = []
            all_ids = []
            
            # bbox, data['width'], data['height']
            #   0          1             2
            if id1_data is not None:
                all_boxes.append(id1_data[0])
                all_ids.append(1)
            if id2_data is not None:
                all_boxes.append(id2_data[0])
                all_ids.append(2)
            
            # 如果有 bbox 就處理
            if len(all_boxes) > 0:
                centers = []
                scales = []
                for box in all_boxes:
                    center, scale = box_to_center_scale(
                        box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]
                    )

                    centers.append(center)
                    scales.append(scale)
                
                # 姿態估計
                pose_preds, confidence = get_pose_estimation_prediction(
                    pose_model, image_pose, centers, scales, transform=pose_transform
                )
                
                # 繪製並儲存
                for keypoints, max_vals, person_id in zip(pose_preds, confidence, all_ids):
                    # 繪製到 skeleton_frame
                    draw_skeleton_and_keypoints(skeleton_frame, keypoints, max_vals)
                    
                    # 儲存 JSON
                    if person_id == 1:
                        json_out = os.path.join(id1_dir, json_filename)
                    else:
                        json_out = os.path.join(id2_dir, json_filename)
                    
                    save_keypoints_json(json_out, f'{video_name}.mp4',
                                    count, fps, keypoints, max_vals)
            
            # 處理缺失的 json
            if id1_data is None:
                save_empty_json(os.path.join(id1_dir, json_filename),
                            f'{video_name}.mp4', count, fps)
            if id2_data is None:
                save_empty_json(os.path.join(id2_dir, json_filename),
                            f'{video_name}.mp4', count, fps)
            
            # 合併並排影片
            combined_frame = np.zeros((frame_h, frame_w * 2, 3), dtype=np.uint8)
            combined_frame[:, :frame_w] = origi_frame
            combined_frame[:, frame_w:] = skeleton_frame
            
            # 寫入影片和截圖
            out.write(combined_frame)
            img_path = os.path.join(img_dir, f'frame_{count:012d}.png')
            cv2.imwrite(img_path, combined_frame)

        cap.release()
        out.release()
        print(f'完成: {video_name}\n')
    
    print('所有影片處理完成！') 

if __name__ == '__main__':
    main() # 為後續鋪路
