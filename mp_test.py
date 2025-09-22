import cv2
import os
import glob
import json
import mediapipe as mp
import numpy as np

def process(input_path, output_path):
    # 初始化
    mp_pose = mp.solutions.pose

    # 受不了了，MP 的點太多太亂了，我決定使用鍵值法去將 MP -> COCO 點，
    # 這樣輸出只會有 COCO 的點位，更好讓我們
    MP_TO_COCO = {
        0: ("nose", 0),         # 鼻子
        2: ("left_eye", 1),     # 左眼
        5: ("right_eye", 2),    # 右眼
        7: ("left_ear", 3),     # 左耳
        8: ("right_ear", 4),    # 右耳
        11: ("left_shoulder", 5),  # 左肩
        12: ("right_shoulder", 6), # 右肩
        13: ("left_elbow", 7),     # 左肘
        14: ("right_elbow", 8),    # 右肘
        15: ("left_wrist", 9),     # 左腕
        16: ("right_wrist", 10),   # 右腕
        23: ("left_hip", 11),      # 左髖
        24: ("right_hip", 12),     # 右髖
        25: ("left_knee", 13),     # 左膝
        26: ("right_knee", 14),    # 右膝
        27: ("left_ankle", 15),    # 左踝
        28: ("right_ankle", 16)    # 右踝
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

    frame_interval = 3 # 每 30 幀取一次
    video_name = os.path.basename(input_path).replace('.mp4', '')
    frame_output = os.path.join(r"D:\rcnn\PyTorch-Object-Detection-Faster-RCNN-Tutorial\output_videos\mp_keypoints",
        video_name
    )
    
    need_video = not os.path.exists(output_path)  # 影片不存在才需要輸出
    need_frames = not os.path.exists(frame_output)  # 資料夾不存在才需要輸出幀
    
    if not need_video and not need_frames:
        print(f"\n  ! {video_name} 的影片和幀資料夾都已存在，跳過處理")
        return False
    
    print(" --- 處理中 ---")
    # 如果需要輸出幀，建立資料夾
    if need_frames:
        os.makedirs(frame_output, exist_ok=True)
        print(f"  • 建立圖片和 json 資料夾：{frame_output}")
    else:
        print(f"  • 圖片和 json 資料夾 {frame_output} 都已存在，但會繼續處理影片")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"無法打開 {input_path}")
        return 
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if need_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))
    else :
        out = None
        print(f"  影片 {os.path.basename(output_path)} 已存在，跳過影片輸出")

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2, # 最嚴格
        smooth_landmarks=True,
        min_detection_confidence=0.2, # 都是越高越嚴格
        min_tracking_confidence=0.1
    )
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # 轉換色彩樣式給 MediaPipe 使用
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb_frame)

        # 建立並排 UI，背景白色
        combined_frame = np.full((h, w * 2, 3), 255, dtype=np.uint8)

        combined_frame[:, :w] = frame # 左邊是原影片
        #            高 寬 儲存格
        # : 全部  ，  w: 從 w 到尾  ， :w 從頭到 w
        # 所以這裡簡單講就是 左右畫一半，然後左邊放原本影片，等等右邊要放骨架影片

        """可依喜好註解掉或保留，這個功能是讓原始照片可以和骨架貼合，更好辨識關鍵點是否在對的位置。"""
        alpha = 0.5  # 透明度，數值越小越淡 (0.0~1.0)
        skeleton_frame = cv2.addWeighted(frame, alpha, np.full_like(frame, 255), 1 - alpha, 0)

        """ 若不要，想要骨架用白色單純背景，和原始圖片分離。"""
        #skeleton_frame = np.full((h, w, 3), 255, dtype=np.uint8) # 右邊是骨架影片

        if res.pose_landmarks:
            if need_frames and frame_count % frame_interval == 0:

                 # 準備儲存JSON資料
                landmarks_data = {
                    "video_name": video_name + ".mp4",
                    "frame_number": frame_count,
                    "timestamp": frame_count / fps,
                    "landmarks": []
                }

                # 提取所有33個關節點資料
                for mp_id, (coco_name, coco_id) in MP_TO_COCO.items():
                    landmark = res.pose_landmarks.landmark[mp_id]
                    
                    landmarks_data["landmarks"].append({
                        "id": coco_id,         # 存入 COCO ID (0-16)
                        "name": coco_name,       # 【新增】存入器官名稱
                        "x": int(landmark.x * w),
                        "y": int(landmark.y * h),
                        "z": landmark.z,
                        "v": landmark.visibility
                    })

                json_path = os.path.join(frame_output, f"frame_{frame_count:012d}.json")
                with open(json_path, 'w', encoding='utf-8') as f: # 只能用中文喔
                    json.dump(landmarks_data, f, indent=2)

                

            # 純畫骨架，沒有客製化
            """
            mp_drawing.draw_landmarks(
                skeleton_frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec = mp_drawing.DrawingSpec
                (
                    color=(0, 255, 0), # 綠色關節點
                    thickness=2,
                    circle_radius=3
                ),
                connection_drawing_spec = mp_drawing.DrawingSpec(
                    color=(0, 0, 0), # 黑色連線
                    thickness=2,
                    circle_radius=2
                )
            )
            """

            """
            mp_drawing.draw_landmarks(
                skeleton_frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec = None,
                connection_drawing_spec = mp_drawing.DrawingSpec(
                    color=(0, 0, 0), # 黑色連線
                    thickness=1,
                    circle_radius=2
                )
            )
            for idx, landmark in enumerate(res.pose_landmarks.landmark):
                # 計算像素座標，因為 mp 預設是標準化，我們需要乘 寬、高 來讓它變成實際位置
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                visibility = landmark.visibility
                
                # 根據信心度決定顏色
                if visibility > 0.8:
                    color = (0, 255, 0)  # 綠色
                elif visibility >= 0.5:
                    color = (0, 165, 255)  # 深橘色
                else:
                    color = (0, 0, 255)  # 紅色
                
                # 畫關節點
                cv2.circle(skeleton_frame, (x, y), 3, color, -1)
                
                # 顯示信心度分數
                cv2.putText(skeleton_frame, 
                           f"{visibility:.2f}", 
                           (x + 5, y - 5),  # 偏右上移一點避免遮擋
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.3,  # 字體大小
                           color, 1)
            """
            
            for c in COCO_SKELETON:
                c_id1, c_id2 = c
                mp_id1 = None
                mp_id2 = None
                for mp_id, (name, coco_id) in MP_TO_COCO.items():
                    if coco_id == c_id1:
                        mp_id1 = mp_id
                    if coco_id == c_id2:
                        mp_id2 = mp_id

                if mp_id1 is not None and mp_id2 is not None:
                    lm1 = res.pose_landmarks.landmark[mp_id1]
                    lm2 = res.pose_landmarks.landmark[mp_id2]
                    
                    x1 = int(lm1.x * w)
                    y1 = int(lm1.y * h)
                    x2 = int(lm2.x * w)
                    y2 = int(lm2.y * h)
                    
                    # 畫連線
                    cv2.line(skeleton_frame, (x1, y1), (x2, y2), (0, 0, 0), 1)

            for mp_id, (name, coco_id) in MP_TO_COCO.items():
                landmark = res.pose_landmarks.landmark[mp_id]
                # 計算像素座標，因為 mp 預設是標準化，我們需要乘 寬、高 來讓它變成實際位置
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                visibility = landmark.visibility
                
                # 根據信心度決定顏色
                if visibility > 0.8:
                    color = (0, 255, 0)  # 綠色
                elif visibility >= 0.5:
                    color = (0, 165, 255)  # 深橘色
                else:
                    color = (0, 0, 255)  # 紅色
                
                # 畫關節點
                cv2.circle(skeleton_frame, (x, y), 3, color, -1)
                
                # 顯示信心度分數
                cv2.putText(skeleton_frame, 
                           f"{visibility:.2f}", 
                           (x + 5, y - 5),  # 偏右上移一點避免遮擋
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.3,  # 字體大小
                           color, 1)

        else:
            cv2.putText(skeleton_frame, "未偵測到人物", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        combined_frame[:, w:] = skeleton_frame
        if need_frames and frame_count % frame_interval == 0 and res.pose_landmarks:
            img_path = os.path.join(frame_output, f"frame_{frame_count:012d}.png")
            cv2.imwrite(img_path, combined_frame)
            print(f"    儲存幀 {frame_count}: {img_path}")
        
        if need_video:
            out.write(combined_frame)
        
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"處理進度: {progress:.1f}%   ({frame_count} / {total_frames})")
        
    cap.release()
    if need_video:
        out.release()
    pose.close()
    if need_video:
        print(f"  \n影片已儲存至 {os.path.basename(output_path)}")
    if need_frames:
        print(f"  \n幀資料儲存至 {frame_output}")
    return True



def main():
    input_dir = r"D:\rcnn\PyTorch-Object-Detection-Faster-RCNN-Tutorial\output_videos"
    output_dir = r"D:\rcnn\PyTorch-Object-Detection-Faster-RCNN-Tutorial\output_videos\mp_keypoints"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已自動建立輸出路徑資料夾 {output_dir}")

    video_ok = os.path.join(input_dir, "test*-*.mp4")
    video_files = glob.glob(video_ok)

    if not video_files:
        print(f" 錯誤: 在 {input_dir} 找不到輸入影片規格是 test00x-x.mp4 的影片")
        return

    print(f"總共找到 {len(video_files)} 個輸入特寫影片\n")

    for idx, input_path in enumerate(video_files, 1): # 從 1 開始編號
        file_name = os.path.basename(input_path)
        output_path = os.path.join(output_dir, file_name)
        
        print(f"  正在處理第 {file_name}\n")

        res = process(input_path, output_path)
        if res == False:
            print(f"  {file_name} 的幀資料夾已存在，跳過，若需重跑要刪掉資料夾。\n")

    print("\n已完成所有待處理的輸入檔")

main()

