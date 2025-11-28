import os
import cv2
import numpy as np
import json
import argparse
from tqdm import tqdm
import imageio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import WildDetDataset
from lib.utils.vismo import render_and_save
from collections import OrderedDict

H36M_KEYPOINTS_17 = {
    0: '髖關節中心',
    1: '右髖',
    2: '右膝',
    3: '右腳踝',
    4: '左髖',
    5: '左膝',
    6: '左腳踝',
    7: '脊椎',
    8: '胸腔',
    9: '鼻子',
    10: '頭頂',
    11: '左肩',
    12: '左手肘',
    13: '左手腕',
    14: '右肩',
    15: '右手肘',
    16: '右手腕'
}

# 我自己的定義輸入輸入
BASE_2D_DETECTION_DIR = r'../ultralytics/output_videos/2d_detections'
# 2d 關鍵點輸入

BASE_VIDEO_DIR = r'../ultralytics/test_videos'
# 影片輸入

BASE_OUTPUT_DIR = r'./output_3d_vidoes'  # 建立一個新資料夾
# 輸出

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    #parser.add_argument('-j', '--json_path', type=str, help='alphapose detection result json path')
    #parser.add_argument('-v', '--vid_path', type=str, help='video path')
    #parser.add_argument('-o', '--out_path', type=str, help='output path')
    parser.add_argument('--pixel', action='store_true', help='align with pixle coordinates')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    opts = parser.parse_args()
    return opts

if __name__ == '__main__':
    opts = parse_args()
    args = get_config(opts.config)

    model_backbone = load_backbone(args)
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()
        print(" 已經用 CUDA 了喵~ \n")
    else:
        print(" 你現在是用 CPU 喔 !\n")

    print('導入 checkpoint 中', opts.evaluate)
    checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)
    model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
    """
    new_state_dict = OrderedDict()

    # 遍歷 checkpoint 中 'model_pos' 的所有參數
    for k, v in checkpoint['model_pos'].items():
        name = k.replace('module.', '') # 移除 'module.' 前綴
        new_state_dict[name] = v      # 將參數存到新的 state_dict 中

    # 載入我們手動修正過的 state_dict
    model_backbone.load_state_dict(new_state_dict, strict=True)
    """

    model_pos = model_backbone
    model_pos.eval()

    # 掃描2D偵測結果的基礎路徑，找到所有影片的資料夾名稱 (如 'test001', 'test002')
    video_folders = [d for d in os.listdir(BASE_2D_DETECTION_DIR) if os.path.isdir(os.path.join(BASE_2D_DETECTION_DIR, d))]


    # 第一層循環：遍歷每個影片
    for video_name in video_folders:
        # 第一次循環，video_name 是 'test001'；處理完後，第二次循環，video_name 變成 'test002'，以此類推。
        video_2d_data_path = os.path.join(BASE_2D_DETECTION_DIR, video_name)

        # 用來存放這支影片所有人的3D姿態結果，因為我要全部一起釋放
        ans = {}

        # 掃描 test00x-x 資料夾，找到所有人物 id 的資料夾(1, 2)
        person_id_folders = [d for d in os.listdir(video_2d_data_path) 
                     if os.path.isdir(os.path.join(video_2d_data_path, d)) 
                     and d.isdigit()] # 確定只有 test00x-1, 2... 的人物，不會把其他多餘的東西放進來。

        # 影片檔名沒有 bbox，所以用 BASE_VIDEO_DIR
        input_video_path = os.path.join(BASE_VIDEO_DIR, f"{video_name}.mp4") # 讀取 有 bbox 的東西


        # 因為同一支影片的所有人共用同一個原始影片檔，所以只需讀取一次影片元資料（fps、尺寸）。
        vid = imageio.get_reader(input_video_path, 'ffmpeg')
        fps_in = vid.get_meta_data()['fps']
        vid_size = vid.get_meta_data()['size']
        vid.close()

        
        output_video_path = os.path.join(BASE_OUTPUT_DIR, f"{video_name}.mp4")
        if os.path.exists(output_video_path):
            print(f"欸欸 {video_name}.mp4 已經存在，跳過處理！\n")
            continue 

        # 建立 json 輸出資料夾
        json_output_folder = os.path.join(BASE_OUTPUT_DIR, video_name)
        os.makedirs(json_output_folder, exist_ok=True)
        
        # 第二層循環：遍歷影片中的每個人物
        for person_id in person_id_folders:
            # test00x 裡面的 1, 2 資料夾
            print(f"--- 正在處理: {video_name} / person_{person_id} ---\n") 

            json_path = os.path.join(video_2d_data_path, person_id)

            testloader_params = {
                    'batch_size': 12,
                    'shuffle': False,
                    'num_workers': 8,
                    'pin_memory': True,
                    'prefetch_factor': 4,
                    'persistent_workers': True,
                    'drop_last': False
            }

            if opts.pixel:
                # Keep relative scale with pixel coornidates
                wild_dataset = WildDetDataset(json_path, clip_len=opts.clip_len, vid_size=vid_size, scale_range=None, focus=opts.focus)
            else:
                # Scale to [-1,1]
                wild_dataset = WildDetDataset(json_path, clip_len=opts.clip_len, scale_range=[1,1], focus=opts.focus)

            test_loader = DataLoader(wild_dataset, **testloader_params)

            results_all = []
            with torch.no_grad():
                for batch_input in tqdm(test_loader):
                    N, T = batch_input.shape[:2]
                    if torch.cuda.is_available():
                        batch_input = batch_input.cuda()
                    if args.no_conf:
                        batch_input = batch_input[:, :, :, :2]
                    if args.flip:    
                        batch_input_flip = flip_data(batch_input)
                        predicted_3d_pos_1 = model_pos(batch_input)
                        predicted_3d_pos_flip = model_pos(batch_input_flip)
                        predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # Flip back
                        predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
                    else:
                        predicted_3d_pos = model_pos(batch_input)
                    if args.rootrel:
                        predicted_3d_pos[:,:,0,:]=0                    # [N,T,17,3]
                    else:
                        predicted_3d_pos[:,0,0,2]=0
                        pass
                    if args.gt_2d:
                        predicted_3d_pos[...,:2] = batch_input[...,:2]
                    results_all.append(predicted_3d_pos.cpu().numpy())

            #results_all = np.hstack(results_all)
            results_all = np.concatenate(results_all)
            # 將這些片段沿著第 0 個維度（批次）拼接起來的正確方法，
            # 它會將 [(1, 243, 17, 3), (1, 243, 17, 3), ...] 變成 (N, 243, 17, 3)，這正是我們想要的長序列。


            if results_all.shape[0] > 0:
                num_clips = results_all.shape[0]
                clip_len = results_all.shape[1] # 將 4D 陣列 (片段數, 幀數, 17, 3) 轉換為 3D 陣列 (總幀數, 17, 3)

                                        # 片段 * 幀數 = 總幀數
                results_all_reshaped = results_all.reshape(-1, 17, 3)

            else:
                # 如果沒有結果，創建一個空的，避免後續報錯
                results_all_reshaped = np.zeros((0, 17, 3))

            # 將這個人的3D姿態結果存入字典
            ans[person_id] = results_all_reshaped
            print(f"<<臨時暫存>> 第 {person_id} 號，處理完成了喵~ \n")

        # 全部都好了，檢查
        if len(ans) == 0:
            print(f"欸欸：{video_name} 沒有任何人，怎麼會這樣!!!!! \n")
            continue
    
        print(f"\n --> 開始為每個人渲染骨架影片 peko peko <--\n")
        
        # 為每個人分別渲染骨架 temp，
        # 一個人一個人來 -> 結束後站存 -> 等等再合併
        skeleton_temp_rec = {} # 儲存「暫存影片的檔案路徑」
        for pid, kp3d in ans.items():
            temp_path = os.path.join(BASE_OUTPUT_DIR, f'temp_video_later_delete_{video_name}_{pid}.mp4')

            print(f"正在畫 第 {pid} 人的骨架")
            render_and_save(kp3d, temp_path, fps=fps_in)
            skeleton_temp_rec[pid] = temp_path
            print(f"第 {pid} 人的骨架繪製完成！\n")


        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        print("\n")
        print(f" 所有 id 人物的資料都好了，準備開始合併結果影片: {video_name}.mp4\n")

        video_reader = imageio.get_reader(input_video_path, 'ffmpeg')
        # 逐幀讀取原始影片（test001.mp4），這樣我們才能拿到左邊要放的畫面

        skeleton_readers = {} # 儲存「讀取暫存影片的 reader 物件」
        for pid, temp_path in skeleton_temp_rec.items():
            skeleton_readers[pid] = imageio.get_reader(temp_path, 'ffmpeg')

        writer = imageio.get_writer(output_video_path, fps=fps_in, codec='libx264') # 輸出
        # 取得總幀數
        #first_person_id = list(ans.keys())[0] # 大家幀數應該都一樣，所以直接取第一個比較快
        #total_frames = ans[first_person_id].shape[0] # 確保總幀數的值

        # 記錄標準尺寸
        standard_h = None
        standard_w = None
        
        try:
            total_frames = video_reader.count_frames()
        except:
            # 如果 count_frames() 不可用，get_length() 通常也可以
            total_frames = video_reader.get_length()

        for frame_idx in tqdm(range(total_frames), desc=f"合併與渲染結果: {video_name}"):
            try:
                original_frame = video_reader.get_data(frame_idx)
            except:
                break
            
            sk2_frames = [] # sk 是 skeloton， sk2 是化妝品
            for pid in sorted(ans.keys()):
                try:
                    skeleton_frame = skeleton_readers[pid].get_data(frame_idx)

                    # 轉換 RGBA → RGB（如果是4通道）
                    if skeleton_frame.shape[2] == 4:
                        skeleton_frame = skeleton_frame[:, :, :3]
                        # [:, :, :3] 意思是：「所有行、所有列、前3個通道」
                    
                    # 調整高度與原始影片一致
                    target_h = original_frame.shape[0] # target_h`：目標高度
                    target_w = int(skeleton_frame.shape[1] * (target_h / skeleton_frame.shape[0]))
                    # skeleton_frame.shape[0]：骨架的原高
                    # skeleton_frame.shape[1]：骨架的原寬
                    # 新高度 / 舊高度 = 新寬度 / 舊寬度
                    skeleton_frame = cv2.resize(skeleton_frame, (target_w, target_h))

                    sk2_frames.append(skeleton_frame)
                except:
                    # 如果讀取失敗，用黑色代替
                    sk2_frames.append(np.zeros_like(original_frame))
            
            # # 水平拼接：[原始影片 | 骨架1 | 骨架2 ]
            if len(sk2_frames) > 0:
                right_half = np.hstack(sk2_frames)
                if standard_h is None:
                    standard_h = original_frame.shape[0]
                    standard_w = right_half.shape[1]
                    print(f"標準尺寸: 高度={standard_h}, 骨架區寬度={standard_w}\n")
                
                #  強制調整到標準尺寸
                if right_half.shape[0] != standard_h or right_half.shape[1] != standard_w:
                    right_half = cv2.resize(right_half, (standard_w, standard_h))

                combined_frame = np.hstack([original_frame, right_half])
            else:
                combined_frame = original_frame
            
            # 建立當前幀的 json 資料
            json_data = {
                "video_name": video_name,
                "frame": frame_idx + 1,
                "keypoints": []
            }
            
            # 遍歷所有人物，收集關鍵點
            for pid in sorted(ans.keys()):
                # 取得這個人在這一幀的 3D 關鍵點 (17, 3)
                if frame_idx < ans[pid].shape[0]:
                    kp3d = ans[pid][frame_idx]  # shape: (17, 3)
                    
                    # 遍歷 17 個關鍵點
                    for kp_id in range(17):
                        keypoint_info = {
                            "id": kp_id,
                            "name": H36M_KEYPOINTS_17[kp_id],
                            "x": float(kp3d[kp_id, 0]),
                            "y": float(kp3d[kp_id, 1]),
                            "z": float(kp3d[kp_id, 2])
                        }
                        json_data["keypoints"].append(keypoint_info)
            
            # 儲存 json 檔案
            json_filename = f"frame_{frame_idx + 1:012d}.json"
            json_path = os.path.join(json_output_folder, json_filename)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            # 寫入合併後的幀
            writer.append_data(combined_frame)


        video_reader.close()
        writer.close()
        for reader in skeleton_readers.values():
            reader.close()
        print(f" {video_name} 完成，準備刪除暫存檔！\n")

        # 刪除暫存骨架影片
        print(f"清理暫存檔案...")
        for temp_path in skeleton_temp_rec.values():
            if os.path.exists(temp_path):
                os.remove(temp_path)
        

        print(f" {video_name} 好了！輸出至 {output_video_path}\n")
