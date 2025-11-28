import argparse
import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn  # 雖然沒用到，但留著也無妨
import cv2
import glob
import json
import numpy as np
import sys
import time
import warnings

# 新增 MMPose/ViTPose 相關的 import (我是參考 top_down_img_demo.py 把相關套件載下來)
from xtcocotools.coco import COCO
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
#init_pose_model: 初始化模型（把模型架構和權重載入記憶體）。
#inference_top_down_pose_model: 進行推論（真正算出關鍵點的核心函式）。
#vis_pose_result: 視覺化（把骨架畫在圖片上）。

from mmpose.datasets import DatasetInfo

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

def get_pose_estimation_prediction(pose_model, image, bboxes, dataset, dataset_info):
    """
    使用 ViTPose/MMPose 進行姿態估計
    """
    # 轉換 bbox 格式
    person_results = []
    for bbox in bboxes: # box = [(x1, y1), (x2, y2)]
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]
        person_results.append({
            'bbox': [x1, y1, x2-x1, y2-y1]  # [x,y,w,h]
        })
    
    # 使用 MMPose API
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image, # 圖片路徑
        person_results, # 剛才打包好的 BBox 列表
        bbox_thr=None,
        format='xywh', # 強調 BBox 格式是 xywh
        dataset=dataset, # 資料集類型
        dataset_info=dataset_info, # 資料集詳細定義
        return_heatmap=False,
        outputs=None
    )
    
    # 解析結果
    coords = []
    confidence = []
    for result in pose_results:
        keypoints = result['keypoints']
        coords.append(keypoints[:, :2]) # 取前 2 行 (columns 0 和 1，也就是 x, y)
                            # 取所有列 (所有 17 個關鍵點)
        confidence.append(keypoints[:, 2:3]) # 取 confidence score
    
    return np.array(coords), np.array(confidence)


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
    bbox = [(x, y), (x + w, y + h)] # box = [(x1, y1), (x2, y2)]
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
        "frame": frame_number,
        "timestamp": frame_number / fps,
        "keypoints": []
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(empty_data, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description='ViTPose 2D pose estimation')
    # general
    #parser.add_argument('--cfg', type=str, required=True)
    #parser.add_argument('--videoFile', type=str, required=True)

    # 模型設定檔 (.py)。
    parser.add_argument('--pose_config', type=str,
                    default=r'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose+_huge_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py',
                    help='ViTPose config file path')
    
    parser.add_argument('--pose_checkpoint', type=str, default=r'weights/vitpose_huge.pth',
                       help='ViTPose checkpoint file path')
    # 模型權重檔(我已經在上面那個模型設定檔準備好了)

    parser.add_argument('--device', default='cuda:0',
                       help='Device used for inference')
    
    parser.add_argument('--input_dir', type=str,
                       default=r'../ultralytics/test_videos')
                        # 這路徑是死的，你們要改可以自己改
    # 因為我有自己的 faster r-cnn 所以我們這邊多加一個設定，bbox 的來源路徑
    parser.add_argument('--bbox_base', type=str,
                       default=r'../ultralytics/output_videos')
    
    #parser.add_argument('--outputDir', type=str, default='/output/')
    parser.add_argument('--output_dir', type=str, 
                       default=r'../ultralytics/output_videos/2d_detections')
    
    #parser.add_argument('--inferenceFps', type=int, default=10)
    #parser.add_argument('--writeBoxFrames', action='store_true')



    args = parser.parse_args()

    return args


def main():
    # transformation

    args = parse_args()

    pose_model = init_pose_model(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device.lower()
    )

    # 取得 dataset 資訊
    dataset = pose_model.cfg.data['test']['type']
    # 檢查 config 檔裡面有沒有定義 dataset_info (例如關鍵點叫啥、連線要連哪裡)
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn('欸欸趕快去設定 dataset_info in the config'
                    'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                    DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

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
        
        # 檢查 1: 是否有對應的資料夾，以免沒有 json folder 還假裝有硬抓，最後甚麼都沒有
        check_folder = os.path.join(args.bbox_base, video_name)
        # 檢查 2: 影片，同樣目的
        check_video = os.path.join(args.bbox_base, os.path.basename(v))

        if not os.path.exists(check_folder) or not os.path.exists(check_video):
            print(f" [跳過] 找不到 YOLO 輸出數據: {video_name} (缺少資料夾或對應影片)")
            print("\n")
            continue
        
        """ 處理輸出路徑 """
        video_output_dir = os.path.join(args.output_dir, video_name) # 資料夾
        output_video_path = os.path.join(args.output_dir, f'{video_name}.mp4') # 影片

        if os.path.exists(video_output_dir) and os.path.exists(output_video_path):
            print(f' 跳過: {video_name}.mp4 及 {video_name} 資料夾已存在，不覆蓋。')
            print("\n")
            continue # 跳過本次迴圈，處理下一個影片

        img_dir = os.path.join(video_output_dir, "img")
        id1_dir = os.path.join(video_output_dir, "1")
        id2_dir = os.path.join(video_output_dir, "2")
        
        os.makedirs(video_output_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(id1_dir, exist_ok=True)
        os.makedirs(id2_dir, exist_ok=True)

        
        cap = cv2.VideoCapture(v)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 輸出影片設定（並排）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, 
                            (frame_w, frame_h))
        
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
            

            image_pose = frame
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
                pose_preds, confidence = get_pose_estimation_prediction(
                                        pose_model, image_pose, all_boxes, dataset, dataset_info)
                
                #print(f"第 {count} 幀: 姿勢推測的格式 shape = {pose_preds.shape}")
                #print(f"  第一個關鍵點(如果是 coco 會是鼻子) = {pose_preds[0][0]}")  # nose
                #print(f"  Frame size = {frame.shape[:2]}")

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
            
            # 寫入影片和截圖
            out.write(skeleton_frame)
            img_path = os.path.join(img_dir, f'frame_{count:012d}.png')
            cv2.imwrite(img_path, skeleton_frame)

        cap.release()
        out.release()
        print(f'完成: {video_name}\n')
    
    print('所有影片處理完成！') 

if __name__ == '__main__':
    main() # 為後續鋪路
