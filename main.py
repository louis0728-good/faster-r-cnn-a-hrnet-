from pathlib import Path
import cv2
import os
import json
import logging
from rich.logging import RichHandler
from read_frame import UICardReader, ThresholdConfig, ZoomableROI
from process_frame import RoundDetector, State
from video_clipper import VideoClipper, group_rounds_for_clipping

MAMA_DIR = Path(__file__).resolve().parent

# === 路徑設定（全部集中在这里管理）===
INPUT_DIR = Path(r"E:\rcnn\fencing_videos")
OUTPUT_DIR = Path(r"E:\rcnn\clipped")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROI_CONFIG_DIR = MAMA_DIR / "roi_configs"
ROI_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# === 其他全域參數 ===
OK_VIDEO_FORMATS = ['.mp4', '.mov', '.mkv']
FRAME_NUM_TO_EXTRACT_FOR_ROI = 0

ROI_TITLES = ["timer", "score_left", "score_right", "period", "board_left", "board_right"]
ROI_TITLES_ZH = {
    "timer": "時間",
    "score_left": "左比分",
    "score_right": "右比分",
    "period": "局",
    "board_left": "左側板得分燈區",
    "board_right": "右側板得分燈區",
}

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)]
)
logger = logging.getLogger(__name__)

class MAIN_PIPELINE:
    def __init__(self, roi_config, threshold_config, input_video_path=None, 
             sample_interval=15.00, output_video_path=str(OUTPUT_DIR)):
        self.input_video_path = input_video_path
        self.roi_config = roi_config
        self.threshold_config = threshold_config
        self.output_video_path = output_video_path
        self.sample_interval = sample_interval

    def run(self):
        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            logger.error(f"無法開啟影片: {self.input_video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 建立輸出目錄
        match = Path(self.input_video_path).stem
        output_match_folder = os.path.join(self.output_video_path, match) # 先建立一個資料夾
        os.makedirs(output_match_folder, exist_ok=True)

        # 初始化各元件
        ui_reader = UICardReader(self.roi_config, self.threshold_config, frame_w, frame_h)
        round_detector = RoundDetector(self.threshold_config, fps, self.sample_interval)
        video_clipper = VideoClipper(self.input_video_path, output_match_folder, fps)

        frame_number = 0
        processed_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % self.sample_interval != 0:
                frame_number += 1
                continue # 每 幾幀取一次 所以中間這些過度的就 跳過  

            print(f"\n{'-'*5} [幀 {frame_number} 處理開始]")
            """
                                                self.last_known_timer: Optional[float] = None
                                                self.last_known_score: List[Optional[int]] = [None, None]
                                                self.last_known_period: Optional[int] = None 
                                                self.prev_ui_visible: bool = False
            """
            info = ui_reader.read_frame(frame, frame_number, fps, last_known_timer=round_detector.last_known_timer, # 上幀 ui 時間
                                                                last_known_score=round_detector.last_known_score,
                                                                last_known_period=round_detector.last_known_period,
                                                                prev_ui_visible=round_detector.prev_ui_visible)
            round_detector.process_frame(info)
            """
            if record is not None:
                # 切割影片
                video_clipper.clip_round(record, record.pause_segments)
                # 儲存回合 JSON
                video_clipper.save_round_json(record)
            """

            # 如果比賽結束
            if round_detector.state == State.MATCH_ENDED:
                logger.info("=== 整場比賽結束，處理下一部影片或已完成所有影片 ===")
                break

            frame_number += 1
            processed_frames += 1

        cap.release()

        round_detector.check_implicit_end(total_frames, fps)
        # 強制結算的回合也要切割和存檔

        groups = group_rounds_for_clipping(round_detector.rounds)
        for group in groups:
            voids, valid = group[:-1], group[-1]
            # 前面的 void：各自留原本那一支（檔名已是 voidXXX），內容不動
            for v in voids:
                video_clipper.clip_round(v, v.pause_segments)   # → voidXXX.mp4
                video_clipper.save_round_json(v)                # → voidXXX.json
            # 收尾回合：普通回合→自己一支；void 組→合併連貫(前無效+後有效，中間非回合剪掉)
            video_clipper.clip_round_group(group)               # → testYYY.mp4
            video_clipper.save_round_json(valid)                # → testYYY.json

        logger.info(f"處理完成！共 {len(round_detector.rounds)} 個回合，輸出至 {self.output_video_path}")
        pass

def get_roi(video_path: str):
    # 20260529 已閱
    """每個影片獨立一份 ROI 設定，但仍會先詢問使用者是否沿用"""
    video_name = Path(video_path).stem
    config_path = ROI_CONFIG_DIR / f"{video_name}.json"
    has_existing = config_path.exists()

    # 不管有沒有設定檔,統一詢問。輸入無效或按 y 但沒設定檔時,重問一次。
    while True:
        ans = input(
            f"是否要沿用 [{video_name}] 設定的 UI 字卡位置？"
            f"(y=沿用既有 / n=重新框選 / s=跳過此影片): "
        ).strip().lower()

        if ans == 'y':
            if not has_existing:
                print(f"  [{video_name}] 沒有既有的 ROI 設定可以沿用,請重新選擇")
                continue  # 重新問

            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"已確定使用 {video_name} 的 ROI 設定")
            return data["rois"], False

        elif ans == 's':
            print(f"使用者選擇跳過 {video_name}\n")
            return None, False

        elif ans == 'n':
            print(f"使用者選擇重新框選 {video_name} 的 ROI...")
            break   # 跳出 while,進入下方框選流程

        else:
            print(f"  無效輸入，請輸入 y / n / s")
            continue   # 重新問

    # 進行框選
    print(f" 開始為 {video_name} 進行 ROI 框選...")
    new_config = customize_roi(str(video_path))

    # 儲存（或覆蓋）這支影片的設定
    data = {"video": video_name, "rois": new_config}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f" 已儲存 {video_name} 的 ROI 設定: {config_path.name}")
    return new_config, True

def customize_roi(video_path):
    roi_dict = {}
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUM_TO_EXTRACT_FOR_ROI) 
    ret, frame = cap.read()
    cap.release()

    if not ret or FRAME_NUM_TO_EXTRACT_FOR_ROI is None:
        raise RuntimeError(f"無法讀取影片: {video_path}")

    for t in ROI_TITLES:
        name = ROI_TITLES_ZH[t]
        selector = ZoomableROI(f"{t}", frame, zh_title=f"請框選: {name}", 
                                                                initial_scale=0.85) # 錨點 視窗初始大小
        rel_coords = selector.run()
        roi_dict[t] = rel_coords
        print(f"已記錄 {name}: {rel_coords}")


    return roi_dict

if __name__ == "__main__":
    thresholds = ThresholdConfig()
    if INPUT_DIR.exists():
        for input_video_path in INPUT_DIR.iterdir():
            if input_video_path.suffix.lower() in OK_VIDEO_FORMATS:
                output_match_folder = OUTPUT_DIR / input_video_path.stem
            
                if output_match_folder.exists():
                    print(f"跳過 {input_video_path.name},輸出資料夾已存在: {output_match_folder.name}")
                    continue
                current_roi, new = get_roi(input_video_path)
                if current_roi is None:
                    # 使用者選擇跳過此影片
                    print(f"已跳過 {input_video_path.name}\n")
                    continue

                print(f">>> 正在處理 {input_video_path.name}")
                pipeline = MAIN_PIPELINE(
                    roi_config=current_roi,
                    threshold_config=thresholds,
                    input_video_path=input_video_path,
                    sample_interval=15,
                )
                pipeline.run()

            else:
                print(f" 跳過非影片檔 {input_video_path.name}")
    else:
        print(" [比賽前處理] 輸入路徑不存在")
