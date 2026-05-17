"""
labels.json 格式：
[
    {"video_name": "test001", "person_id": 1, "label_upper": 0, "label_lower": 3},
    {"video_name": "test001", "person_id": 2, "label_upper": 1, "label_lower": 5},
    ...
]

輸出 PKL 結構：
{
    'split': {
        'xsub_train': ['test001_p1', 'test002_p2', ...],
        'xsub_val':   ['test001_p2', 'test003_p1', ...] # 影片名稱
    },
    'annotations': [
        {
            'frame_dir': 'test001_p1',
            'label_upper': 0,
            'label_lower': 3,
            'img_shape': (1080, 1920),
            'original_shape': (1080, 1920),
            'total_frames': 150,
            'keypoint': np.array(...),       # (1, T, 17, 2)
            'keypoint_score': np.array(...)  # (1, T, 17)
        },
        ...
    ]
}
"""
import os
import glob
import json
import numpy as np
import argparse
import random
import mmengine

MAMA_dir = os.path.dirname(os.path.abspath(__file__))
DETECTIONS_DIR = os.path.normpath(
    os.path.join(MAMA_dir, '..', 'ultralytics', 'output_videos', '2d_detections')
)
OUTPUT_DIR = os.path.join(MAMA_dir, 'pkl')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_person_jsons(json_dir): # 這裡先把所以人的 json 都累積起來
    # 讀取某一個人資料夾裡所有 .json，回傳 (video_name, total_frames, xy, score)
    files = sorted(glob.glob(os.path.join(json_dir, '*.json')))
    if not files:
        raise FileNotFoundError(f'找不到任何 json: {json_dir}')

    #frames = []
    xys = []
    scores = []
    video_name = None
    img_w = None  
    img_h = None

    for ff in files:
        with open(ff, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if video_name is None:
            video_name = data.get('video_name', os.path.basename(json_dir))
            img_w = data.get('width')
            img_h = data.get('height')

        #frame_idx = int(data['frame'])
        kps = data.get('keypoints', [])
        # 沒偵測到人 / 空關鍵點：直接補 0
        if len(kps) == 0:
            xy = np.zeros((17, 2), dtype=np.float32)
            sc = np.zeros((17,), dtype=np.float32)
        else:
            #kps = sorted(kps, key=lambda x: x['id'])

            xy = np.array([[kp['x'], kp['y']] for kp in kps], dtype=np.float32)
            sc = np.array([kp['v'] for kp in kps], dtype=np.float32)

            if xy.shape != (17, 2):
                raise ValueError(f'{ff} keypoint shape 是 {xy.shape}, 與預期不符 (17, 2)')
            if sc.shape != (17,):
                raise ValueError(f'{ff} score shape 是 {sc.shape}, 與預期不符 (17,)')

        #frames.append(frame_idx)
        xys.append(xy)
        scores.append(sc)

    # 依 frame 排序後，補成連續時間軸
    #order = np.argsort(frames)
    #frames = np.array(frames)[order]
    # 排序後應該怎麼重排 frames 照舊順序，但實際讀取資料看 order 順序，不影響原本 frames 的排版
    xys = np.stack(xys, axis=0)    # (Tp, 17, 2)
    scores = np.stack(scores, axis=0) # (Tp, 17)
    total_frames = len(xys)

    return video_name, total_frames, xys, scores, (img_h, img_w)

# 讀取一部影片的雙人骨架，拆成每人獨立資料
def load_video_persons(video_dir):
    person_dirs = []
    for pid in [1, 2]:
        pdir = os.path.join(video_dir, str(pid)) # pdir = test00x 地址 + 1/2
        if os.path.isdir(pdir):
            person_dirs.append((pid, pdir))

    if not person_dirs:
        raise FileNotFoundError(f'{video_dir} 下找不到 1/ 或 2/ 子資料夾')

    results = []
    ref_video_name = None
    ref_total_frames = None

    for pid, pdir in person_dirs: # pid = 1, 2, pdir = test00x 地址 + 1/2
        video_name, total_frames, xys, scores, img_shape = load_person_jsons(pdir)

        if ref_video_name is None:
            ref_video_name = video_name
            ref_total_frames = total_frames
        else:
            if video_name != ref_video_name:
                raise ValueError(f'影片名稱不一致: {ref_video_name} vs {video_name}')
            if total_frames != ref_total_frames:
                raise ValueError(f'幀數不一致: {ref_total_frames} vs {total_frames}')

        keypoint = xys[np.newaxis, ...]          # (1, T, 17, 2)
        keypoint_score = scores[np.newaxis, ...]  # (1, T, 17)

        results.append({
            'person_id': pid,
            'video_name': ref_video_name,
            'total_frames': total_frames,
            'keypoint': keypoint,
            'keypoint_score': keypoint_score,
            'img_shape': img_shape,
        })

    return results

def build_train_pkl(label_json_path, output_pkl_path,
                    train_ratio=0.8, seed=42): # 0.8 代表 80% 訓練 20 驗證
    with open(label_json_path, 'r', encoding='utf-8') as f:
        label_list = json.load(f)
    """
    把 labels.json 讀進來，轉成一個字典，key 是 ('test001', 1) 
    這種 tuple，value 是上下半身標籤。這樣等一下讀到某部影片的某個人時，
    可以用 label_map[('test001', 1)] 立刻查到他的標籤。如果同一組 key 出現兩次就報錯。
    """
    #  labels.json: {"video_name": "test001", "person_id": 1, "label_upper": 0, "label_lower": 3},
    label_map = {}
    for item in label_list:
        key = (item['video_name'], item['person_id'])
        if key in label_map:
            raise ValueError(f'標籤重複: {key}') # 同影片名同 id
        label_map[key] = { 
            'label_upper': int(item['label_upper']),
            'label_lower': int(item['label_lower']),
        }

    video_dirs = sorted([
        d for d in glob.glob(os.path.join(DETECTIONS_DIR, '*'))
        if os.path.isdir(d)
    ]) #　去 2d_detections\ 底下找所有子資料夾（每個資料夾代表一部影片），按名稱排序。

    if not video_dirs:
        raise FileNotFoundError(f'在 {DETECTIONS_DIR} 下找不到任何影片資料夾')

    annotations = []
    all_frame_dirs = []

    for vdir in video_dirs: # 每個影片資料夾
        video_name = os.path.basename(vdir)

        if not (os.path.isdir(os.path.join(vdir, '1'))
                and os.path.isdir(os.path.join(vdir, '2'))):
            print(f'[跳過] {video_name}（缺少 1/ 或 2/ 子資料夾）')
            continue

        persons = load_video_persons(vdir) # 會包含每部影片的每 1, 2 個人
        """
        person 裡面大概會長這樣
        results.append({
            'person_id': pid,
            'video_name': ref_video_name,
            'total_frames': total_frames,
            'keypoint': keypoint,
            'keypoint_score': keypoint_score,
            'img_shape': img_shape,
        })
        """
        for person_data in persons: # 跑遍同一部影片的兩個人
            pid = person_data['person_id'] # 來自 keypoints 的
            key = (video_name, pid) # (test00x, 1/2) # 這個才是要去找 labels.json

            if key not in label_map:
                print(f'[跳過] {video_name}_p{pid}（在 labels.json 中找不到標籤）')
                continue

            # label_map[(item['video_name'], item['person_id'])] 　key = (video_name, pid)
            """ 
            label_map[key] = { 
                'label_upper': int(item['label_upper']),
                'label_lower': int(item['label_lower']),
            }
            """
            labels = label_map[key] 
            # label_upper, label_lower 這邊的 key 是從 kp 那個取出來 ，現在查表對應的 label.json 的 upper\lower 標籤
            frame_dir = f'{video_name}_p{pid}'

            anno = {
                'frame_dir': frame_dir,
                'label_upper': labels['label_upper'],
                'label_lower': labels['label_lower'],
                'img_shape': person_data['img_shape'],
                'original_shape': person_data['img_shape'],
                'total_frames': person_data['total_frames'],
                'keypoint': person_data['keypoint'],
                'keypoint_score': person_data['keypoint_score'],
            }
            annotations.append(anno)
            all_frame_dirs.append(frame_dir) # f'{video_name}_p{pid}'

    if not annotations:
        raise RuntimeError('沒有任何有效樣本，請確認骨架資料夾和 labels.json')

    print(f'\n共 {len(annotations)} 筆樣本')

    random.seed(seed) # 每次執行這個腳本時，隨機打亂的順序都會一模一樣
    indices = list(range(len(all_frame_dirs))) 
    # 把總共的幾筆資料(單人 f'{video_name}_p{pid}')做 list 變成 0 ~ indices-1
    random.shuffle(indices) # 打亂

    split_point = int(len(indices) * train_ratio)
    train_dirs = [all_frame_dirs[i] for i in indices[:split_point]]
    val_dirs = [all_frame_dirs[i] for i in indices[split_point:]]

    print(f'訓練集: {len(train_dirs)} 筆, 驗證集: {len(val_dirs)} 筆')

    data = {
        'split': {
            'xsub_train': train_dirs,
            'xsub_val': val_dirs,
        },
        'annotations': annotations,
    }

    os.makedirs(os.path.dirname(output_pkl_path) or '.', exist_ok=True)
    mmengine.dump(data, output_pkl_path)
    print(f'\nPKL 已儲存: {output_pkl_path}')

    for i, anno in enumerate(annotations[:3]):
        print(f'  樣本 {i}: frame_dir={anno["frame_dir"]}, '
              f'upper={anno["label_upper"]}, lower={anno["label_lower"]}, '
              f'kp_shape={anno["keypoint"].shape}, '
              f'score_shape={anno["keypoint_score"].shape}, '
              f'total_frames={anno["total_frames"]}, '
              f'img_shape={anno["img_shape"]}')
        # 印出前 3 幀的第 0 個人的 keypoint 和 score
        kp = anno['keypoint']    # (1, T, 17, 2)
        sc = anno['keypoint_score']  # (1, T, 17)
        for t in range(min(3, kp.shape[1])):
            print(f'    幀 {t} keypoint (前5點): {kp[0, t, :5, :]}')
            print(f'    幀 {t} score    (前5點): {sc[0, t, :5]}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='建立訓練用 PKL')
    parser.add_argument('--labels', type=str,
                    default=os.path.join(MAMA_dir, 'labels.json'),
                    help='labels.json 的路徑')
    parser.add_argument('--output', type=str,
                        default=os.path.join(OUTPUT_DIR, 'train_val.pkl'),
                        help='輸出 PKL 路徑')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='訓練集比例 (預設 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子 (預設 42)')
    args = parser.parse_args()

    build_train_pkl(
        label_json_path=args.labels,
        output_pkl_path=args.output,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )