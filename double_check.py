import cv2
import json
import os
import glob
import numpy as np

def working(video_path, outputs_dir):

    video_name = os.path.basename(video_path) 
    video_base = os.path.splitext(video_name)[0]

    ur9ay = os.path.join(outputs_dir, video_base)
    
    # 分別定義 ID1 和 ID2 的 JSON 路徑
    json_folder_id1 = os.path.join(ur9ay, '1')
    json_folder_id2 = os.path.join(ur9ay, '2')

    if not os.path.exists(json_folder_id1):
        print(f"找不到對應的 json 資料夾: {json_folder_id1}")
        return
    
    if not os.path.exists(json_folder_id2):
        print(f"找不到對應的 json 資料夾: {json_folder_id2}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟影片: {video_name}")
        return
    
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    output_path = os.path.join(ur9ay, f"{video_base}_bbox驗證.mp4")
    if os.path.exists(output_path):
        print(f"驗證影片已存在，跳過: {output_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        # 定義要處理的 ID 和對應的資料夾
        id_check = {
            "1": json_folder_id1,
            "2": json_folder_id2
        }

        # 遍歷 ID1 和 ID2
        for _, json_folder in id_check.items():
            json_filename = f"frame_{frame_count:012d}.json"
            json_path = os.path.join(json_folder, json_filename)

            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)

                    bbox = json_data.get('bbox', [])
                    if len(bbox) == 4:
                        x, y, w, h = map(int, bbox)

                        # 統一使用綠色繪製
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

                except Exception as e:
                    print(f"讀取或處理 json 檔案失敗 {json_path}: {e}")

                out.write(frame)
    
    cap.release()
    out.release()
    
    print(f"驗證影片已儲存: {output_path}")

def main():
    base_dir = r'D:\rcnn\PyTorch-Object-Detection-Faster-RCNN-Tutorial\output_videos'
    temp = os.path.join(base_dir, 'test*.mp4')

    video_files = glob.glob(temp)
    
    if not video_files:
        print("沒有找到任何影片檔案")
        return
    
    for v in video_files:
        working(v, base_dir)

    print("所有驗證皆完成！")

main()
