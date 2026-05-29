import cv2, os, json, logging, numpy as np, re, torch, subprocess
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import field
from dataclasses import dataclass, field
from enum import Enum, auto
from rich.logging import RichHandler
from collections import Counter
import easyocr
    
if(torch.cuda.is_available()):
    print(" 正在使用 cuda 喔")
else:
    print(" 懷疑正在使用 cpu ")

OCR_ENGINE = "easyocr"
MAMA_DIR = Path(__file__).resolve().parent
#INPUT_DIR = MAMA_DIR / 'test_videos'
INPUT_DIR = Path(r"E:\rcnn\fencing_videos")   # 我會把東西放在 e 槽
#OUTPUT_DIR = MAMA_DIR / 'output_videos'
OUTPUT_DIR = Path(r"E:\rcnn\clipped")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OK_VIDEO_FORMATS = ['.mp4', '.mov', '.mkv']
ROI_CONFIG_DIR = MAMA_DIR / "roi_configs"
ROI_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

FRAME_NUM_TO_EXTRACT_FOR_ROI = 0

ROI_TITLES = ["timer", "score_left", "score_right", "period", "board_left", "board_right"]
ROI_TITLES_ZH = {
    "timer": "時間",
    "score_left": "左比分",
    "score_right": "右比分",
    "period": "局",
    "board_left": "左側板",
    "board_right": "右側板",
}

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)]
)
logger = logging.getLogger("clip_video")

class ThresholdConfig:
    # 計時器停頓判定 
    timer_pause_frames: int = 60
    # 調大就更寬 調小更嚴 我是覺得最少一秒，實驗後調整

    # 調寬越寬鬆 調窄更嚴格
    lower_red1 = (0, 120, 140) # 也可以考慮 lower_red1 = (0, 100, 100), or 120
    upper_red1 = (10, 255, 255)

    lower_red2 = (170, 120, 140)
    upper_red2 = (179, 255, 255)

    # 調寬寬鬆 調窄嚴格
    green_range: Tuple[int, int] = (40, 85)

    # 飽和度下限：過濾掉白色/灰色區域
    # 調大嚴格 調小寬鬆
    light_sat_min: int = 170 # 原 120

    # 亮度
    # 調小更寬鬆 調大更嚴格
    bright_min: int = 180

    # 紅/綠色佔整個 roi 的比例門檻
    # 調小更寬鬆 調大更嚴格
    rg_color_ratio: float = 0.05

    # 調小更寬鬆 調大更嚴格
    board_color_change: float = 25

    # 調小更寬鬆 調大更嚴格
    #ui_visible_rate: float = 0.40 # 與參考幀的相似度低於此值 字卡不可見

    # 這邊單位都是 s 
    passivity_seconds: float = 50.00 # 比賽開始幾秒了(判定消極比賽用)
    #passivity_stall_seconds: float = 50.0  # 單次停頓多久算消極(影片中途切入的保險)
    stall_seconds: float = 1.00 # 消極判定時，計時器需停頓超過此秒數才確認
    ui_hasnt_show_up_min: float = 3.0 # 最後一個回合結束後，UI 字卡消失後 x 秒內沒出現，代表比賽結束了
    ui_timer_lag: float = 5.0 # 如果 ui 延遲出現可以容忍幾秒
    ocr_timer_confidence: float = 0.50
    ocr_other_confidence = 0.53
    too_late_to_win_thres = 0.2
    votes_rate = 0.5
    clip_jump_timer_delta: float = 4.5   # 時間突然減少超過幾秒就視為剪輯跳變
    # 得分後,導播慢半拍才按暫停造成的多餘倒數,容許的「相鄰」幀數範圍
    # 預設 = sample_interval 的 1 倍多一點,只擋緊鄰得分的那一兩幀
    director_lag_frames: int = 20
    timer_rise_limit: float = 3.0   # 回合中 timer 往上跳超過幾秒就視為誤讀,沿用前一幀


@dataclass
class FrameInfo:
    frame_number: int = 0
    timestamp: float = 0.0  
    timer_value: Optional[float] = None
    score_left: Optional[int] = None
    score_right: Optional[int] = None
    light_left: bool = False   
    light_right: bool = False
    board_color_left: Optional[Tuple[float, float, float]] = None  
    board_color_right: Optional[Tuple[float, float, float]] = None
    period: Optional[int] = None        
    ui_visible: bool = False  

class State(Enum):
    WAITING_FOR_START = auto()       # 等同一局內的下一回合
    IN_ROUND = auto() 
    TIMER_STALLED = auto()
    WAITING_FOR_NEW_PERIOD = auto() 
    MATCH_ENDED = auto()             # 整場比賽已結束
    
def set_window_title(cv2_name, zh_title): # 改中文
    try:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, cv2_name)
        if hwnd:
            ctypes.windll.user32.SetWindowTextW(hwnd, zh_title)
    except Exception:
        pass

class ZoomableROI:
    def __init__(self, window_name, image, zh_title="", initial_scale=0.5):
        self.window_name = window_name
        self.zh_title = zh_title
        self.orig_img = image.copy()
        self.scale = initial_scale
        self.drawing = False
        self.roi = None
        self.ix, self.iy = -1, -1
        self.cur_x, self.cur_y = -1, -1

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_evt)
        set_window_title(self.window_name, self.zh_title)

    def show(self):
        # 根據 scale 縮放影像
        h, w = self.orig_img.shape[:2]
        scaled_w, scaled_h = int(w * self.scale), int(h * self.scale)
        display = cv2.resize(self.orig_img, (scaled_w, scaled_h))

        if self.drawing: # 畫 roi
            cv2.rectangle(display, (self.ix, self.iy), (self.cur_x, self.cur_y), (0, 255, 0), 2)
        elif self.roi:
            x, y, rw, rh = self.roi
            cv2.rectangle(display, (x, y), (x + rw, y + rh), (0, 255, 0), 2)

        cv2.imshow(self.window_name, display)

    def mouse_evt(self, event, x, y, flags, param):
        h, w = self.orig_img.shape[:2]
        scaled_w, scaled_h = int(w * self.scale), int(h * self.scale)
        safe_x = max(0, min(x, scaled_w - 1))
        safe_y = max(0, min(y, scaled_h - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = safe_x, safe_y
            self.cur_x, self.cur_y = safe_x, safe_y
            self.roi = None
            self.show()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.cur_x, self.cur_y = safe_x, safe_y
                self.show()

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.cur_x, self.cur_y = safe_x, safe_y
            # 確保左上角與右下角順序正確
            x_min, x_max = min(self.ix, self.cur_x), max(self.ix, self.cur_x)
            y_min, y_max = min(self.iy, self.cur_y), max(self.iy, self.cur_y)
            rw = max(1, x_max - x_min)
            rh = max(1, y_max - y_min)

            self.roi = (x_min, y_min, rw, rh)
            self.show()

    def run(self):
        self.show()
        while True:
            key = cv2.waitKey(1) & 0xFF
            # Enter(13) or Space(32) 確認
            if key == 13 or key == 32:
                if self.roi is not None:
                    break
                else:
                    print("請先框選區域！")
        cv2.destroyWindow(self.window_name)

        # 記錄用相對座標紀錄
        h, w = self.orig_img.shape[:2]
        scaled_w, scaled_h = int(w * self.scale), int(h * self.scale)
        rx, ry, rw, rh = self.roi # 真實的 roi 座標
        
        return (rx / scaled_w, ry / scaled_h, rw / scaled_w, rh / scaled_h) 
        # 回傳比例，不希望說因為 roi 時視窗有調整大小，框選的區域在實際推論時失真

# 讀 ui 字卡
class UICardReader:
    def __init__(self, roi_config, threshold_config, frame_width: int, frame_height: int):
        self.roi = roi_config
        self.threshold_config = threshold_config
        self.w = frame_width
        self.h = frame_height

        if easyocr is not None:
            self.reader = easyocr.Reader(["en"], gpu=True, verbose=False)
        else:
            self.reader = None
            logger.warning("打 pip install easyocr")

        # 快取上一幀的計分板顏色
        self.prev_board_color_left: Optional[Tuple[float, float, float]] = None
        self.prev_board_color_right: Optional[Tuple[float, float, float]] = None

    
    def _crop(self, frame: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
        # 切 roi 區域
        rx, ry, rw, rh = roi
        x1 = int(rx * self.w)
        y1 = int(ry * self.h)
        x2 = int((rx + rw) * self.w)
        y2 = int((ry + rh) * self.h)
        return frame[y1:y2, x1:x2]

    def read_timer(self, frame, last_timer: Optional[float] = None):
        crop = self._crop(frame, self.roi["timer"])
        text, candidates = self.ocr_text(crop, allowlist="0123456789:.", title="時間", ocr_thresh=self.threshold_config.ocr_timer_confidence) # 錨點 時間點
        timer_value = self.parse_timer_text(text, last_timer)

        was_repaired = False
        # 如果正常解析失敗，但 OCR 有讀到文字 → 啟動保守救援
        if timer_value is None:
            timer_value = self._repair_timer_if_possible(candidates, last_timer)
            if timer_value is not None:
                was_repaired = True
                logger.debug(f"  ├─ [OCR 時間] 修補成功: 候選={candidates} → {timer_value}")
        return timer_value, was_repaired

    def read_score_left(self, frame) -> Optional[int]:
        crop = self._crop(frame, self.roi["score_left"])
        text, _ = self.ocr_text(crop, allowlist="0123456789", title="左選手成績", ocr_thresh=self.threshold_config.ocr_other_confidence)
        return self.parse_score_text(text)

    def read_score_right(self, frame) -> Optional[int]:
        crop = self._crop(frame, self.roi["score_right"])
        text, _ = self.ocr_text(crop, allowlist="0123456789", title="右選手成績", ocr_thresh=self.threshold_config.ocr_other_confidence)
        return self.parse_score_text(text)

    def read_period(self, frame, last_period: Optional[int] = None):
        crop = self._crop(frame, self.roi["period"])
        text, _ = self.ocr_text(crop, allowlist="1234/", title="局", ocr_thresh=self.threshold_config.ocr_other_confidence)
        period_value = self.parse_period_text(text)
    
        was_repaired = False
        if period_value is None and last_period is not None:
            period_value = last_period   # 沿用上一幀
            was_repaired = True
        return period_value, was_repaired

    def _preprocess_variants(self, crop: np.ndarray, target_h: int = 80) -> List[np.ndarray]:
        """
        原 ROI ─┬─ 變體 A (灰階+CLAHE) ──→ EasyOCR ──→ "9"
        ├─ 變體 B (Otsu)        ──→ EasyOCR ──→ "9"  
        ├─ 變體 C (反向 Otsu)   ──→ EasyOCR ──→ "4"  ← 這個變體被字型騙了
        └─ 變體 D (自適應)      ──→ EasyOCR ──→ "9"
        """
        h, w = crop.shape[:2]
        
        if h < target_h:
            scale = target_h / h
            crop = cv2.resize(crop, (int(w * scale), target_h), 
                            interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # CLAHE — 處理計分板背景漸層光暈
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        variants = [enhanced]  # 變體 1:強化後灰階(保留最多細節)
        
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)  # 變體 2:標準 Otsu
        
        _, otsu_inv = cv2.threshold(enhanced, 0, 255, 
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        variants.append(otsu_inv)  # 變體 3:反向(處理深底淺字 / 淺底深字未知的情況)
        
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 31, 10
        )
        variants.append(adaptive)  # 變體 4:自適應(處理區域光線不均)
        
        return variants

    def ocr_text(self, crop: np.ndarray, allowlist: str, title:str, ocr_thresh:float) -> Tuple[str, List[str]]:
        if self.reader is None:
            return ""
        
        variants = self._preprocess_variants(crop)
        list_of_text_conf = []

        for v in variants:
            results = self.reader.readtext(
                v,
                allowlist=allowlist,
                detail=1, # 細節    
                paragraph=False, # 確保讀到的字體獨立
                text_threshold=0.60, # 最後字體判定門檻
                low_text=0.30,  # 入口門檻
                mag_ratio=1.5,
                width_ths=0.50, # 水平方向合併門檻提高
                height_ths=0.80, # 垂直
            )
            if not results:
                continue

            # 該變體最有把握的那個讀數
            best_choices = max(results, key=lambda r: r[2])
            _, text, conf = best_choices
            text = text.strip()
            if text:
                list_of_text_conf.append((text, conf))

        if not list_of_text_conf:
            return "", []
        
        # 多數決投票
        counter = Counter(text for text, _ in list_of_text_conf)
        winner, vote_count = counter.most_common(1)[0]

        # 平手時拿信心當裁判
        top_two = counter.most_common(2)
        if len(top_two) >= 2 and top_two[0][1] == top_two[1][1]:
            # 找信心總和較高的那個
            candidates = [t for t, c in top_two[:2]]
            sum_conf = {t: sum(c for txt, c in list_of_text_conf if txt == t) for t in candidates}
            winner = max(sum_conf, key=sum_conf.get)
            vote_count = counter[winner]

        # 一致性門檻:得票要過半才採信
        n = len(list_of_text_conf)
        if vote_count / n < self.threshold_config.votes_rate:
            logger.debug(f"  ├─ [OCR {title}] 投票分歧 {dict(counter)},本幀不可靠")
            return "", [t for t, _ in list_of_text_conf]

        logger.debug(f"  ├─ [OCR {title}] 多數決:'{winner}' (贏家票數: {vote_count} / 總票數: {n})")
        return winner, [t for t, _ in list_of_text_conf]


    @staticmethod
    def parse_timer_text(text: str, last_known_timer: Optional[float]) -> Optional[float]:
        # m: ss 
        if not text:
            return None
        
        # 錨點 解析時間
        text = (text.replace(" ", "")
                    .replace(";", ":")
                    .replace(",", ".")
                    .replace("l", "1")
                    .replace("I", "1")
                    .replace("o", "0")
                    .replace("O", "0"))

        if len(text) == 4 and text.isdigit():
            if text[1] in ('3', '2'):                    # 第二個字元如果是 3 or 2
                minutes = int(text[0])
                seconds = int(text[2:])           # 取後面兩位當秒數
                if 0 <= minutes <= 3 and 0 <= seconds <= 59:
                    total = minutes * 60 + seconds
                    return float(total)
            
        # 當 OCR 讀到 3.00 / 300 / 3:00 且上一刻 timer 很小時 → 直接當成 180 秒
        # 我把 3.0 刪掉了，因為有些比賽影片會以 3.0 作為 3 秒，而非 0:03。就是那麼機掰
        # 這樣做是為了避免系統誤會以為 倒數 3 秒時把它當作 新局開始。
        if text in ("3.00", "300", "3:00") or (len(text) == 3 and text.startswith("3") and text[1:].isdigit()):
            if last_known_timer is None or last_known_timer <= 10.0:   # 上一刻是休息/結束狀態
                logger.debug(f"[parse_timer_text] 偵測到新局開始的 '3.00'，強制解析為 180 秒")
                return 180.0
            
        # ★ 新增: 嚴格格式檢查
        # 合法 timer 格式只有 "M.SS" / "MM.SS" / "M:SS" / "MM:SS" 等
        # 整體必須能完整匹配,否則視為 OCR 異常
        if not re.fullmatch(r"\d{1,2}[.:]\d{1,2}", text):
            logger.debug(f"[parse_timer_text] '{text}' 格式畸形,交給 _repair_timer_if_possible")
            return None
    
        # 標準格式 M:S 分:秒
        match = re.search(r"(\d{1}):(\d{2})", text)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            return minutes * 60.0 + seconds

        # 情況 2：OCR 讀出小數點 (可能真的是小數，也可能是冒號被誤認)
        # 秒數小數格式（如 8.2 4.35 0.17）
        match = re.search(r"(\d{1})\.(\d{1,2})", text)
        if match:
            # 這樣 9.1 就是 9.1，不會變成 9.01
            possible_sss = float(match.group(0)) # 當作 秒.毫秒 (0.33)

            digit1 = int(match.group(1))
            digit2 = int(match.group(2))

            # 擊劍一局最多 3 分鐘。如果第一位數 >=4 或是 ==3 但是後面的尾數並非0(代表已經超過極限了，可以判定應該不是 分:秒  
            # (例如 4.35, 3.32、8.20)
            # 那可以肯定應該是 秒:毫秒 s:ss
            if digit1 >= 4 or digit1 ==3 and digit2 != 00:
                return possible_sss
            
            # 第一位數是 0, 1, 2, 3 (電腦會讀成例如: 0.33、0.46、3.00)
            # 有兩種可能代表，以 0.33 舉例，搞不好其實是: 0:33 秒 或 真的 0.33 毫秒 
            possible_ms = float(digit1 * 60 + digit2)    # 當作 分:秒 (0.33 -> 真實 0:33 但 ocr 讀錯)

            # 如果有歷史時間，看哪一個數字比較合理 (離上一幀比較近)
            if last_known_timer is not None:
                if abs(last_known_timer - possible_ms) <= abs(last_known_timer - possible_sss):
                    return possible_ms
                else:
                    return possible_sss
            else:
                # 剛開局沒有歷史紀錄的防呆機制：
                # 第一幀極大機率是正常的 分:秒 (包含 3:00、2:15、0:45 等等)
                # 擊劍通常在 10 秒以下才會顯示小數點。所以如果後面的數字 >= 10，優先當成分:秒
                if digit1 >= 1 or (digit1 == 0 and digit2 >= 10):
                    return possible_ms
                return possible_sss
            
        # 情況 3：小數點被 ocr 吃掉的純數字防呆，例如: 7.1 變成 71
        # 只能是純數字，不能有任何其他東西
        match = re.search(r"^(\d+)$", text.strip())
        if match:
            num_str = match.group(1)
            num = int(num_str)
            
            if last_known_timer is not None:
                # 假設 OCR 把 7.1 讀成了 71，或是 9.0 讀成 90
                # 如果把數字除以 10 後，離上一幀 (例如 7.2) 差距小於 1 秒，就用他
                if len(num_str) == 2 and abs(last_known_timer - (num / 10.0)) < 1.0:
                    return num / 10.0

        return None
    
    @staticmethod
    # claude 寫的 但是我腦袋 算力用完了 改天再檢查對不對。
    def _repair_timer_if_possible(candidates: List[str], 
                               last_known_timer: Optional[float]) -> Optional[float]:
        """
        當 parse_timer_text 解不出 timer 時的救援。
        用「上一幀的 (分, 秒) pair」當 anchor,從候選字串中找最合理的 timer。
        僅在分:秒區段 (last_known_timer > 10) 啟用,避免跟 parse_timer_text 的
        秒.毫秒邏輯打架。
        """
        if last_known_timer is None or last_known_timer <= 10.0:
            return None
        if not candidates:
            return None

        # 建立 expected pairs:同秒、-1、-2、-3 秒
        # 這範圍要夠寬吸收卡頓,但要夠窄擋住真實剪輯跳變
        expected = {}  # (分, 秒) -> 該 pair 對應的總秒數
        base = int(last_known_timer)
        for delta in range(0, 4):
            s = max(0, base - delta)
            expected[(s // 60, s % 60)] = float(s)

        votes = Counter()
        for text in candidates:
            digits = re.sub(r'\D', '', text.strip())
            if not digits:
                continue

            # 嘗試多種「(分, 秒)」分割
            attempts = set()
            if len(digits) == 3:
                attempts.add((int(digits[0]), int(digits[1:3])))
            elif len(digits) == 4:
                # 後兩位當秒(冒號被讀成 2 的情境,如 '1224' → (1,24))
                attempts.add((int(digits[0]), int(digits[2:4])))
                # 中間兩位當秒(備用)
                attempts.add((int(digits[0]), int(digits[1:3])))
            else:
                continue

            # 過濾不合理 pair
            attempts = {(m, s) for (m, s) in attempts if 0 <= m <= 3 and 0 <= s <= 59}

            # 每個 candidate 至多投一票
            for pair in attempts:
                if pair in expected:
                    votes[expected[pair]] += 1
                    break

        if not votes:
            return None

        winner_sec, _ = votes.most_common(1)[0]
        return winner_sec

    @staticmethod
    def _repair_score_if_possible(
        raw_sl: Optional[int],
        raw_sr: Optional[int],
        last_known_score: List[Optional[int]],
    ) -> Tuple[Optional[int], Optional[int], bool]:
        """
        任一側 OCR 沒讀到 → 沿用上一幀。
        不看 light、不看 timer (因為亮燈時 OCR 本來也不會即時更新,沿用等同預期行為)。
        回傳: (修補後 sl, 修補後 sr, 是否有修補)
        """
        if raw_sl is not None and raw_sr is not None:
            return raw_sl, raw_sr, False   # 兩邊都有,不用修
        
        if last_known_score[0] is None and last_known_score[1] is None:
            return raw_sl, raw_sr, False   # 沒歷史值可抄
        
        new_sl = raw_sl if raw_sl is not None else last_known_score[0]
        new_sr = raw_sr if raw_sr is not None else last_known_score[1]
        repaired = (raw_sl is None or raw_sr is None) and new_sl is not None and new_sr is not None
        return new_sl, new_sr, repaired

    @staticmethod
    # 錨點 成績 戰績
    def parse_score_text(text: str) -> Optional[int]:
        if not text:
            return None
        match = re.search(r"(\d{1,2})", text.strip())
        if match:
            return int(match.group(0))
        return None
    
    @staticmethod
    def parse_period_text(text: str) -> Optional[int]:
        if not text:
            return None
        text = text.replace("|", "/").replace("\\", "/").replace(" ", "").replace(":", "/").replace(".", "/")

        if len(text) == 3 and text.isdigit():
            # 2x3 → 視為 2/3，取第一位當作局數
            period = int(text[0])
            logger.debug(f"[OCR 局] 偵測到三位數字 '{text}'，視為 {period}/3")
            return period
        

        match = re.search(r"(\d)\s*/\s*\d", text)
        if match:
            return int(match.group(1)) # 回傳哪一局
        
        return None

    # 有沒有亮紅/綠燈得分
    def detect_light_left(self, frame: np.ndarray) -> bool:
        crop = self._crop(frame, self.roi["board_left"])
        return self.is_light_on_left(crop)

    def detect_light_right(self, frame: np.ndarray) -> bool:
        crop = self._crop(frame, self.roi["board_right"])
        return self.is_light_on_right(crop)

    def is_light_on_left(self, crop: np.ndarray) -> bool:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        r_lo1 = self.threshold_config.lower_red1
        r_lo2 = self.threshold_config.lower_red2
        r_up1 = self.threshold_config.upper_red1
        r_up2 = self.threshold_config.upper_red2

        mask1 = cv2.inRange(hsv, r_lo1, r_up1)
        mask2 = cv2.inRange(hsv, r_lo2, r_up2)
        mask_red = cv2.bitwise_or(mask1, mask2)
        red_ratio = cv2.countNonZero(mask_red) / crop.size

        s_min = self.threshold_config.light_sat_min
        v_min = self.threshold_config.bright_min
        sat_bright_mask = cv2.inRange(hsv, (0, s_min, v_min), (179, 255, 255))
        final_mask = cv2.bitwise_and(mask_red, sat_bright_mask)
        red_ratio = cv2.countNonZero(final_mask) / crop.size
        return red_ratio > self.threshold_config.rg_color_ratio

    def is_light_on_right(self, crop: np.ndarray) -> bool:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        s_min = self.threshold_config.light_sat_min
        v_min = self.threshold_config.bright_min
        g = self.threshold_config.green_range
        
        mask = cv2.inRange(hsv, (g[0], s_min, v_min), (g[1], 255, 255))
        green_ratio = cv2.countNonZero(mask) / crop.size
        return green_ratio > self.threshold_config.rg_color_ratio
    
    # 取得「計分板背景色」
    def get_board_color_left(self, frame: np.ndarray) -> Tuple[float, float, float]:
        crop = self._crop(frame, self.roi["board_left"])
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return tuple(np.mean(hsv.reshape(-1, 3), axis=0))

    def get_board_color_right(self, frame: np.ndarray) -> Tuple[float, float, float]:
        crop = self._crop(frame, self.roi["board_right"])
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return tuple(np.mean(hsv.reshape(-1, 3), axis=0))
    
    def read_frame(self, frame: np.ndarray, frame_number: int, 
                            fps: float, last_timer: Optional[float] = None, last_known_score=None, 
                                                last_period=None, prev_ui_visible=True) -> FrameInfo:
        info = FrameInfo()
        info.frame_number = frame_number
        info.timestamp = frame_number / fps if fps > 0 else 0.0

        # 各欄位讀取(timer/period 現在回傳 tuple)
        info.timer_value, timer_repaired = self.read_timer(frame, last_timer)
        info.score_left = self.read_score_left(frame)
        info.score_right = self.read_score_right(frame)
        info.period, period_repaired = self.read_period(frame, last_period)

        # === 新增：更清楚的 score 修補 log ===
        raw_sl = info.score_left
        raw_sr = info.score_right

         # 新增: 沒讀到的分數沿用上一幀
        info.score_left, info.score_right, score_repaired = self._repair_score_if_possible(
            info.score_left, info.score_right, last_known_score or [None, None]
        )
        if score_repaired:
            repaired_msg = []
            if raw_sl is None:
                repaired_msg.append(f"ocr 失效，左側根據前幀推論 (None → {info.score_left})")
            if raw_sr is None:
                repaired_msg.append(f"ocr 失效，右側根據前幀推論 (None → {info.score_right})")
            
            logger.debug(
                f"[幀 {frame_number}] [score 修補] {' + '.join(repaired_msg)}"
            )

        # ★ 補償計數投票 ★
        # 分開計算「左右分數」各算一票,所以總票數最多 4 票(timer/score_left/score_right/period)
        repair_votes = 0
        if timer_repaired:
            repair_votes += 1
        if period_repaired:
            repair_votes += 1
        if raw_sl is None and info.score_left is not None:   # 左分數是補出來的
            repair_votes += 1
        if raw_sr is None and info.score_right is not None:  # 右分數是補出來的
            repair_votes += 1
            
        missing = []
        if info.timer_value is None:
            missing.append("timer")
        if info.score_left is None:
            missing.append("score_left")
        if info.score_right is None:
            missing.append("score_right")
        if info.period is None:
            missing.append("period")

        # 原本的邏輯，照用
        if missing:
            info.ui_visible = False
            logger.debug(f"[幀 {frame_number}] 補償後仍缺 {' + '.join(missing)}，判定 UI 消失")
            return info
        
        # 規則2:補償票數過半(>=3),且前一幀已經是 UI 消失 → 這幀極不可信,判消失
        too_many_repairs = (repair_votes >= 2 and not prev_ui_visible)

        # 在可能還是 ui 消失期間 ocr 可能會修復某些讀到的雜訊以為是數字，但其實是錯的
        if too_many_repairs:
            info.ui_visible = False
            logger.warning(
                f"[幀 {frame_number}] 4 項中有 {repair_votes} 項靠補償,且前幀 UI 已消失,"
                f"判定此幀不可信 → UI 消失（避免補償掰出假畫面）"
            )
            return info


        info.ui_visible = True
        info.light_left = self.detect_light_left(frame)
        info.light_right = self.detect_light_right(frame)
        info.board_color_left = self.get_board_color_left(frame)
        info.board_color_right = self.get_board_color_right(frame)

        logger.debug(
                f"  └─ [讀取總結] timer={RoundDetector.format_timer(info.timer_value)}, "
                f"score=[{info.score_left}, {info.score_right}], "
                f"period={info.period}, light=[{info.light_left}, {info.light_right}]"
            )
        
        return info

@dataclass
class RoundRecord:
    # 紀錄每回合的 json test00x
    filename: str = ""
    period: int = 1
    round_number: int = 1
    time_startStamp_for_MAMA: str = "00:00:00.00" # 對於原始影片來說取自哪個時間點到哪個時間點
    time_endStamp_for_MAMA: str = "00:00:00.00"    
    start_frame: int = 0           
    end_frame: int = 0           
    timer_start: str = "0.00" # 影片中局的開始 time
    timer_end: str = "0.00" # 影片中局的結束 time
    score_before: List[int] = field(default_factory=lambda: [0, 0])
    score_after: List[int] = field(default_factory=lambda: [0, 0])
    winner: str = "none"       # "left" / "right" / "both" / "none"
    end_type: str = "win"       # "win" / "double_win" / "passivity" / "period_end"
    #bonus_match: bool = False
    match_end: bool = False
    #gap_before: bool = False           
    #gap_after: bool = False        
    #confidence: str = "high"    
    details: List[str] = field(default_factory=list) # 後面 yolo 接著用
    score_invalidated: bool = False
    is_disputed: bool = False           # 新增: 此回合經過 reconcile 修正
    dispute_type: str = ""              # 新增: "void"(假回合) / "wrong_winner"(誤判方向) / ""

    # 僅在 match_end=True 時寫入
    match_winner: Optional[str] = None
    #match_win_method: Optional[str] = None

    def to_dict(self) -> dict:
        """轉為輸出用的 dict，排除內部欄位"""
        d = {
            "filename": self.filename,
            "period": self.period,
            "round_number": self.round_number,
            "time_startStamp_for_MAMA": self.time_startStamp_for_MAMA,
            "time_endStamp_for_MAMA": self.time_endStamp_for_MAMA,
            "timer_start": self.timer_start,
            "timer_end": self.timer_end,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "winner": self.winner,
            "end_type": self.end_type,
            #"bonus_match": self.bonus_match,
            "match_end": self.match_end,
            #"gap_before": self.gap_before,
            #"gap_after": self.gap_after,
            #"confidence": self.confidence,
            "details": self.details,
            "score_invalidated": self.score_invalidated,
            "is_disputed": self.is_disputed,
            "dispute_type": self.dispute_type,
        }
        if self.match_end:
            d["match_winner"] = self.match_winner
            #d["match_win_method"] = self.match_win_method
        return d
    
class RoundDetector:
    def __init__(self, threshold_config: ThresholdConfig, fps: float):
        self.thresh = threshold_config
        self.fps = fps
        self.state = State.WAITING_FOR_START
        self.last_known_timer: Optional[float] = None
        self.last_known_score: List[Optional[int]] = [None, None]
        self.no_score_added_count: float = 0.0  # 從上一次得分之後，到現在已經過了多久
        self.no_score_added_count_before_last_round: float = 0.0   # 專門只針對那種上回合判錯，不算是一個回合時 消極比賽倒數會繼續數的功能
        self.prev_timer: Optional[float] = None  # 上一幀的計時器讀數
        self.timer_stall_count: int = 0 # 計時器未變化的連續幀數
        self.timer_stall_start_frame: int = 0 # 停頓開始的幀號
        self.light_seen_during_stall_left: bool = False   # 本次停頓期間,左燈是否曾經亮過
        self.light_seen_during_stall_right: bool = False  # 本次停頓期間,右燈是否曾經亮過
        self.board_changed_seen_during_stall: bool = False  # 本次停頓期間,板色是否曾明顯變化
        self._init_score_buf: dict = {'left': [], 'right': []} # 初始幀緩衝

        # 回合
        self.current_period: int = 1
        self.round_count: int = 0
        self.current_round_start_frame: int = 0
        self.current_round_start_time: float = 0.0
        self.current_round_timer_start: Optional[float] = None
        self.timer_at_last_score: Optional[float] = None
        self.prev_round_score: List[int] = [0, 0]
        self.rounds: List[RoundRecord] = []
        #self.last_low_timer: Optional[float] = None
        self.after_period_end: bool = False
        self.last_seen_period: Optional[int] = None 
        # 這兩個都是為了去準備說 局結束間的1分鐘到計時休息時間 那個不能算回合
        self.last_record_needs_rewrite: bool = False  # 上一個 record 被校正過,要重寫 JSON

        # 字卡
        self.prev_ui_visible: bool = False
        self.ui_disappeared_frame: int = 0
        self.last_score_frame: int = -1   # 上次「得分類」回合結束的幀號,-1 代表還沒發生過
        #self.gap_pending: bool = False  # 字卡消失期間是否有缺口
        self.is_first_ui_appearance: bool = True # 為了第一幀
        self.passivity_count: int = 0

        # 記分板顏色
        self.prev_board_color_left: Optional[Tuple[float, float, float]] = None
        self.prev_board_color_right: Optional[Tuple[float, float, float]] = None
        self.board_color_at_stall_left: Optional[Tuple[float, float, float]] = None
        # 計時器暫停時 可能的左右邊得分
        self.board_color_at_stall_right: Optional[Tuple[float, float, float]] = None
        self.frames_since_light_left: int = 30  # 距離上次左燈亮過了幾幀
        self.frames_since_light_right: int = 30
        # 記錄最後一次亮燈的「絕對幀號」
        self.last_light_left_frame: int = -9999
        self.last_light_right_frame: int = -9999
        # 新增：每個回合開始時的固定 board_color 基準
        self.reference_board_color_left: Optional[Tuple[float, float, float]] = None
        self.reference_board_color_right: Optional[Tuple[float, float, float]] = None


        # 一般中途暫停就不產生獨立回合
        # 自動裁切掉暫停期間的無效畫面，在暫停前與恢復後之間無縫拼接
        self.pause_start_and_end: List[Tuple[int, int]] = []  # (暫停 start, end)

    def _update_score_validated(self, info: FrameInfo, side: str):
        """
        第一道防線：初始化防呆  偷偷收集連續 3 幀的答案。
        如果連 3 幀裡面有 2 幀都讀到 0，它才會正式宣布「好，初始分數是 0」。

        第二道防線：物理約束防呆 反正就是比分不會暴增或減少
        """
        from collections import Counter
        
        idx = 0 if side == 'left' else 1
        new_value = info.score_left if side == 'left' else info.score_right
        if new_value is None:
            return
        
        last = self.last_known_score[idx]
        
        # Case A:前幾幀,取眾數
        if last is None:
            buf = self._init_score_buf[side]
            buf.append(new_value)
            if len(buf) >= 3:
                most_common, count = Counter(buf[-3:]).most_common(1)[0] # 讀取緩衝
                if count >= 2:  # 連 3 幀裡至少 2 幀同值才採信
                    self.last_known_score[idx] = most_common
                    logger.debug(f"[初始化] {side} 分數鎖定為 {most_common} "
                                f"(前 3 幀讀取結果: {buf[-3:]})")
            return
        

        # 新增: 回合間期 (STALLED/WAITING) OCR 延遲是正常現象,默默忽略倒退
        # 不更新 last_known_score、不印 log、不修補 info.score
        # 留 info.score 是原始 OCR 值給 reconcile 用
        if self.state in (State.TIMER_STALLED, State.WAITING_FOR_START, State.WAITING_FOR_NEW_PERIOD):
                return
            
        # Case B:已有歷史,驗證 delta
        delta = new_value - last
        
        if delta < 0:
            # 倒退不接受 — 分數的物理約束
            logger.debug(f"[幀 {info.frame_number}] {side} 分數倒退 ({last} → {new_value}), 沿用上一幀")
            if side == 'left':
                info.score_left = last           # ← 新增:把 info 也改回上一幀
            else:
                info.score_right = last
            return

        if delta >= 2:
            # 跨度可疑:正常取樣下不會跳超過 1
            logger.debug(f"[幀 {info.frame_number}] {side} 分數跨度可疑 "
                        f"({last} → {new_value}), 沿用上一幀分數")
            # last_known_score 不變 (這行 self.last_known_score[idx] = last 是多餘的,可拿可留)
            if side == 'left':
                info.score_left = last           # ← 新增
            else:
                info.score_right = last
            return
    
        self.last_known_score[idx] = new_value

    def process_frame(self, info: FrameInfo) -> Optional[RoundRecord]:
        should_update_timer = True

        # 新增:回合中 timer 突然往上跳很多 → 誤讀,沿用前一幀的數字繼續
        if (self.state == State.IN_ROUND
                and info.timer_value is not None
                and self.last_known_timer is not None
                and info.timer_value > self.last_known_timer + self.thresh.timer_rise_limit):
            logger.warning(
                f"[幀 {info.frame_number}] 回合中 timer 反而往上跳 "
                f"{self.format_timer(self.last_known_timer)} → {self.format_timer(info.timer_value)}"
                f"，視為誤讀 → 沿用前一幀 {self.format_timer(self.last_known_timer)}"
            )
            info.timer_value = self.last_known_timer
            
        # 新增：UI 持續顯示下的剪輯跳變偵測
        if self.prev_ui_visible and info.ui_visible:
            is_real_jump, should_update_timer = self._detect_clip_jump(info)
            if is_real_jump:
                record = self.handle_clip_jump(info)
                if record is not None:
                    self.prev_ui_visible = True
                    return record   # 直接讓 MAIN_PIPELINE 去 clip + save json
                
        if not info.ui_visible and self.prev_ui_visible:
            # 上一幀還看得到，這一幀突然不見了 記錄消失瞬間的狀態
            if self.state in (State.IN_ROUND, State.TIMER_STALLED):
                logger.debug(f"[幀 {info.frame_number}] UI 字卡消失，記錄最後已知狀態")
                self.ui_disappeared_frame = info.frame_number

            # 無論在什麼狀態，只要 UI 不見了，標籤就必須撕掉！  
            self.prev_ui_visible = False
            return None

         # 上一幀不可見，這一幀突然出現了，會包括處理第一幀
        if not self.prev_ui_visible and info.ui_visible:
            if self.is_first_ui_appearance:
                logger.debug(f"[幀 {info.frame_number}] 首次偵測到 UI 字卡，開始追蹤")
                self.is_first_ui_appearance = False  # 撕掉標籤，以後就是「重新出現」了
            else:
                if self.state in (State.IN_ROUND, State.TIMER_STALLED):
                    record = self.handle_ui_reappear(info)
                    if record is not None:
                        self.prev_ui_visible = True  # 別忘了更新
                        if info.score_left is not None and info.score_right is not None:
                            self.last_known_score = [info.score_left, info.score_right] 
                            # 只有在雙方分數都出來 才可以記錄。
                        return record
                else:
                    # 局間休息、等待新局、比賽結束期間的 UI 波動 → 一律忽略
                    logger.debug(f"[幀 {info.frame_number}] 休息/等待/新回合開始第一幀，期間 UI 重現")
                    self.prev_ui_visible = True
                    #return None

        # ★ 新增:UI 持續不可見期間,雜訊幀一律不更新任何跨幀狀態 ★
        # 只有「可見 → 不可見」的轉折幀(上面已處理)和「不可見 → 可見」的重現幀(上面已處理)
        # 會走到這裡。若這幀依然不可見,代表還在空白期,直接凍結狀態返回。
        if not info.ui_visible:
            self.prev_ui_visible = False
            logger.debug(f"[幀 {info.frame_number}] UI 持續消失中,凍結狀態(不更新 timer/score/燈號記憶)")
            return None

        self.prev_ui_visible = info.ui_visible

        # 更新燈號計數
        if info.light_left:
            self.last_light_left_frame = info.frame_number
        if info.light_right:
            self.last_light_right_frame = info.frame_number
        self.frames_since_light_left = info.frame_number - self.last_light_left_frame
        self.frames_since_light_right = info.frame_number - self.last_light_right_frame

        # 更新 last_known 狀態(變成目前狀態，等等會更新目前狀態，因為所謂的目前狀態應該是指上一幀)
        # 時間
        if info.timer_value is not None and should_update_timer:
            if self.state == State.WAITING_FOR_NEW_PERIOD and self.after_period_end:
                # 只有當讀到的時間非常接近 3:00 (大於 175 秒)，代表新局真的開始了，才放行更新
                if info.timer_value >= 175.0:
                    self.last_known_timer = info.timer_value
            else:
                self.last_known_timer = info.timer_value

        # 更新成績
        self._update_score_validated(info, 'left')
        self._update_score_validated(info, 'right')

        # 更新計分板顏色
        if info.board_color_left is not None:
            self.prev_board_color_left = info.board_color_left
        if info.board_color_right is not None:
            self.prev_board_color_right = info.board_color_right

        

        if self.state == State.WAITING_FOR_START: # 上一回合已經結束（或還沒開始），等計時器再次開始倒數的那段空檔。
            return self.waiting_for_start(info)

        elif self.state == State.IN_ROUND:
            return self.in_round(info)

        elif self.state == State.TIMER_STALLED:
            return self.timer_stalled(info)

        elif self.state == State.WAITING_FOR_NEW_PERIOD:
            return self.waiting_for_new_period(info)

        elif self.state == State.MATCH_ENDED:
            return None  # 比賽已結束，不再處理

        return None

    def waiting_for_start(self, info: FrameInfo) -> Optional[RoundRecord]:
        """
        等待回合開始。
        1. 整場比賽的第一劍
        2. 同一局內的接續回合（最常見）
        3. 換局開場（2/3、3/3）
        4. 字卡消失後重新出現（在 handle_ui_reappear 處理）
        """
        if info.timer_value is None:
            return None

        if self.prev_timer is not None and info.timer_value < self.prev_timer:

            """ 同理 都先 ban ，因為暫停時間改為 60 幀
            # ★ 新增:導播遲滯攔截 ★
            # 若剛得分完,且這一幀「緊鄰」得分幀(在 director_lag_frames 內),
            # 代表這次倒數極可能是導播慢半拍才按暫停造成的多餘倒數,而非新回合。
            # 忽略這一幀:不開新回合、不跑 reconcile,但要更新 prev_timer 讓下一幀能正常比較。
            if self.last_score_frame >= 0:
                frames_since_score = info.frame_number - self.last_score_frame
                if 0 < frames_since_score <= self.thresh.director_lag_frames:
                    logger.info(
                        f"[幀 {info.frame_number}] 偵測到緊鄰得分({self.last_score_frame})的倒數 "
                        f"timer={self.format_timer(info.timer_value)},距得分僅 {frames_since_score} 幀,"
                        f"判定為導播慢半拍補按暫停,忽略此幀,不開新回合"
                    )
                    # 用掉這次機會,避免下一幀又被擋(下一幀就讓它正常判定回合開始)
                    self.last_score_frame = -1
                    self.prev_timer = info.timer_value   # ← 關鍵:更新 prev_timer
                    return None
            """
            
            # 第一局第一回合開始
            # 字卡顯示 1/3、分數 0:0，且計時器從 3:00 附近開始倒數
            # 允許轉播字卡慢幾秒才浮現
            is_first_match = (
                self.round_count == 0
                and info.period == 1
                and info.score_left == 0
                and info.score_right == 0
                and info.timer_value >= (180.0 - self.thresh.ui_timer_lag)
            )

            # 上一回合結束後（計時器處於停頓狀態），計時器數值再次開始往下掉
            is_continuation = (
                self.round_count > 0
                and self.prev_timer is not None
                and info.timer_value < 180 # 原 < self.prev_timer
            )

            # 影片有可能從後面的回合或段落剪輯，這裡是假設對於系統來說第一次，但是對於真實比賽來說已經進行一陣子的影片
            # 只要 round_count == 0，且不符合完美開局的條件 (有分數、或是局數 >1、或是時間已經明顯小於 3:00)
            # 因為是中途切入，我們「不需要等時間掉」，直接當作回合已經在進行中，立刻強制開局
            has_score = (info.score_left is not None and info.score_left > 0) or \
                        (info.score_right is not None and info.score_right > 0)
            not_first_period = (info.period is not None and info.period > 1)
            tolerate_ui_lag = (info.timer_value < 180.0 - self.thresh.ui_timer_lag)

            is_first_period_for_system_but_not_for_MAMA = (
                self.round_count == 0 and tolerate_ui_lag)

            if is_first_match or is_continuation or is_first_period_for_system_but_not_for_MAMA:
                self.begin_round(info)

        self.prev_timer = info.timer_value
        return None

    def in_round(self, info: FrameInfo) -> Optional[RoundRecord]:
        # 回合進行中。持續追蹤計時器是否停頓。
        if info.timer_value is None:
            return None
        
        # 消極比賽
        #  僅用於消極判定的 60 秒門檻。計時器停頓時不累加，得分時歸零
        if self.prev_timer is not None and info.timer_value < self.prev_timer:
            elapsed = self.prev_timer - info.timer_value
            self.no_score_added_count += elapsed


        """ 優先因為我發現 0.00 抓 1 秒有點太難了 ，現在改為只要時間歸零那就結算"""
        if info.timer_value == 0.0:
            if self.timer_stall_start_frame == 0:
                self.timer_stall_start_frame = info.frame_number
                self.board_color_at_stall_left = info.board_color_left
                self.board_color_at_stall_right = info.board_color_right
            
            self.state = State.TIMER_STALLED
            # 立刻丟給 timer_stalled 處理 (如果是壓哨得分，它會先被 get_point 攔截；如果沒有，就會順利進 times_up)
            return self.timer_stalled(info)

        # 偵測計時器停頓
        """ timer 值一變就立刻把 start 重設為當前幀 
        只要這一幀的 timer 跟上一幀相同，就累計 (current - start)，
        """
        if self.prev_timer is None or info.timer_value != self.prev_timer:
        # 第一次讀，或 timer 值剛剛改變 → 立刻重設停頓起點
            self.timer_stall_start_frame = info.frame_number
            self.timer_stall_count = 0
            # 記錄這一刻的計分板顏色，作為日後變化比對基準
            self.board_color_at_stall_left = info.board_color_left
            self.board_color_at_stall_right = info.board_color_right
            self.light_seen_during_stall_left = False
            self.light_seen_during_stall_right = False
            self.board_changed_seen_during_stall = False
        else:
            # timer 跟上一幀相同 → 累計
            self.timer_stall_count = info.frame_number - self.timer_stall_start_frame
            # 新增: 持續觀察,只要這次停頓期間任何一幀亮過燈,就記住
            if info.light_left:
                self.light_seen_during_stall_left = True
            if info.light_right:
                self.light_seen_during_stall_right = True

            # 板色照燈號的模式:停頓期間變過一次就記住
            if info.board_color_left and self.reference_board_color_left:
                if np.sqrt(sum((c - p) ** 2 for c, p in zip(
                        info.board_color_left, self.reference_board_color_left))) > self.thresh.board_color_change:
                    self.board_changed_seen_during_stall = True
            if info.board_color_right and self.reference_board_color_right:
                if np.sqrt(sum((c - p) ** 2 for c, p in zip(
                        info.board_color_right, self.reference_board_color_right))) > self.thresh.board_color_change:
                    self.board_changed_seen_during_stall = True

            if self.timer_stall_count >= self.thresh.timer_pause_frames:
                self.state = State.TIMER_STALLED
                logger.debug(f"[幀 {info.frame_number}] 計時器停頓大於設定閾值，進入判定")
                return self.timer_stalled(info)
        self.prev_timer = info.timer_value

    def is_period_end(self, info: FrameInfo) -> bool:

        if info.timer_value is not None and abs(info.timer_value) == 0.0: # timer_value 單位是總秒數
            return True
        
        return False

    def timer_stalled(self, info: FrameInfo) -> Optional[RoundRecord]:
        """
        計時器停頓要做啥
        1. 分數改變 ＋ 得分燈亮 ＋ 計分板顏色有明顯變化 = 得分交鋒
        2. 計時器 = 0:00 = 該局結束（period_end） OR  可能系統沒辨識到 0:00 ，下一次讀到已經開始休息時間1 分鐘的倒數了。
        3. no_score_added_count ≥ 60 秒 ＋ 停頓 X 秒 ＋ 計分板顏色無明顯變化 = 消極比賽
        4. 以上皆否 = 一般中途暫停（不產生獨立回合，做拼接處理）
        # 5.（可能會做）分數改變 ＋ 得分燈未亮 = 罰分得分
        """
        if info.timer_value is None:
            return None
        if self.timer_stall_start_frame != 0:
            self.timer_stall_count = info.frame_number - self.timer_stall_start_frame
        # 可以知道這段暫停到底持續了多久 timer_stall_count

        score_left = info.score_left if info.score_left is not None else self.last_known_score[0]
        score_right = info.score_right if info.score_right is not None else self.last_known_score[1]


        #light_on = info.light_left or info.light_right # 原本
        effective_light_left = info.light_left or self.light_seen_during_stall_left
        effective_light_right = info.light_right or self.light_seen_during_stall_right
        light_on = effective_light_left or effective_light_right

        # 計分板顏色是否有明顯變化
        board_color_changed_left = False
        board_color_changed_right = False
        if info.board_color_left and self.reference_board_color_left:
            diff_l = np.sqrt(sum((c - p) ** 2 for c, p in zip(
                info.board_color_left, self.reference_board_color_left)))
            board_color_changed_left = diff_l > self.thresh.board_color_change

        if info.board_color_right and self.reference_board_color_right:
            diff_r = np.sqrt(sum((c - p) ** 2 for c, p in zip(
                info.board_color_right, self.reference_board_color_right)))
            board_color_changed_right = diff_r > self.thresh.board_color_change

        board_color_changed = board_color_changed_left or board_color_changed_right or self.board_changed_seen_during_stall
        #stall_seconds = self.timer_stall_count / self.fps

        # 1. 得分燈亮 ＋ 計分板顏色有明顯變化 + 計時器停頓(且確保是專屬於因為得分而暫停) = 得分交鋒
        if light_on and board_color_changed and (self.prev_timer is None or 
                                                 info.timer_value == self.prev_timer):
            # 把記憶到的燈號狀態寫回 info,讓 get_point 用到
            info.light_left = effective_light_left
            info.light_right = effective_light_right
            return self.get_point(info)

 
        # 該局時間到: 以偵測到 0:00 出現在字卡上為準
        if self.is_period_end(info):
            if self.current_period < 3:
                self.after_period_end = True
                logger.debug(f"[幀 {info.frame_number}] 偵測到 0:00，開啟局間休息忽略模式")
            return self.times_up(info, score_left, score_right)

        # 消極比賽
        #三者同時：
        # - no_score_added_count 累計達 60 秒
        # - 計時器明確停頓超過 1 秒
        # - 分數無變化
        #
        # 輔助判定（必要）：
        # 計分板背景顏色未出現得分時的那種明顯顏色變化。
        # 消極:從回合開始,計時鐘已經倒數滿 50 秒,且現在處於停頓
        # 另一種可能 暫停太久(其實是消極比賽 只是因為影片關係可能影片看起來並沒有滿一分鐘)，所以原本 60 改成 50 了
        if (self.current_round_timer_start is not None
                and info.timer_value is not None
                and (self.current_round_timer_start - info.timer_value) >= self.thresh.passivity_seconds
                #and stall_seconds >= self.thresh.stall_seconds
                and not light_on):
            logger.debug(
                f"[幀 {info.frame_number}] 消極判定觸發: "
            )
            return self.end_round_passivity(info, score_left, score_right)

        # 回合暫停，不會單獨算一回合 會銜接中間空閒片段
        # 錨點 中途暫停
        # 我再想這邊是否可以把 not light_on  給刪掉，因為如果有 light on 應該會被前面的 get_point 觸發? 
        # 我先刪掉試試 原本: if (not light_on and self.timer_stall_start_frame != 0):
        if self.prev_timer is not None and info.timer_value < self.prev_timer:
            if (self.timer_stall_start_frame != 0):
                logger.debug(
                    f"[幀 {info.frame_number}] 計時器恢復倒數,"
                    f"將 ({self.timer_stall_start_frame} → {info.frame_number}) 記為一般中途暫停"
                )
                self.pause_start_and_end.append((self.timer_stall_start_frame, info.frame_number))

            self.state = State.IN_ROUND
            self.timer_stall_count = 0
            self.timer_stall_start_frame = info.frame_number
            self.prev_timer = info.timer_value
            return None

        # 停頓持續中,沒任何條件滿足 → 維持 TIMER_STALLED 繼續等
        self.prev_timer = info.timer_value
        return None

    def waiting_for_new_period(self, info: FrameInfo) -> Optional[RoundRecord]:
        # 等待新局開始。換局開場（2/3、3/3 或延長賽 4/3）
        if self.after_period_end:
            # 只有看到接近 3:00 且 period 有更新時才解除
            if info.timer_value is not None and \
                            info.timer_value >= 175.0 and self.last_seen_period is not None \
                                            and info.period is not None \
                                            and info.period > self.last_seen_period:
                self.after_period_end = False
                self.last_seen_period = info.period

                logger.info(f"[幀 {info.frame_number}] 局間休息模式解除，進入第 {info.period} 局")

                self.state = State.WAITING_FOR_START
                self.prev_timer = info.timer_value
                #self.begin_round(info)
        return None


    def begin_round(self, info: FrameInfo):
        self._reconcile_previous_record(info)

        # 記錄新回合的起始狀態
        if info.period is not None:
            self.current_period = info.period
            self.last_seen_period = info.period

        self.round_count += 1
        self.current_round_start_frame = info.frame_number
        self.current_round_start_time = info.timestamp

        # 上回合若為假回合(void),保留消極錨點,不重設
        last_was_void = bool(self.rounds) and self.rounds[-1].dispute_type == "void"
        if not last_was_void:
            self.current_round_timer_start = self.prev_timer if self.prev_timer is not None else info.timer_value
        else:
            logger.debug(
                f"[幀 {info.frame_number}] 上回合為假回合,保留消極錨點 "
                f"真實有效回合的開始秒數 ={self.current_round_timer_start}"
            )

        self.prev_round_score = [
            info.score_left if info.score_left is not None else self.last_known_score[0],
            info.score_right if info.score_right is not None else self.last_known_score[1],
        ]
        self.last_known_score = list(self.prev_round_score)   

        # 記錄本回合的 board_color 固定基準
        self.reference_board_color_left = None
        self.reference_board_color_right = None

        self.pause_start_and_end = []
        self.state = State.IN_ROUND
        self.timer_stall_count = 0
        self.timer_stall_start_frame = info.frame_number
        logger.info(
            f"[幀 {info.frame_number}] === 回合 {self.round_count} 開始 === "
            f"局={self.current_period}, 計時器={self.format_timer(info.timer_value)}, "
            f"比分={self.prev_round_score}"
        )

        if (self.reference_board_color_left is None
                and not info.light_left and not info.light_right
                and info.board_color_left is not None
                and info.board_color_right is not None):
            self.reference_board_color_left = info.board_color_left
            self.reference_board_color_right = info.board_color_right
            logger.debug(f"[幀 {info.frame_number}] 該回合面板板色鎖定，將用於後續比較得分")

    def _reconcile_previous_record(self, info: FrameInfo):
        """
        新回合開始時,根據 OCR 觀察校驗上回合 record:
        - 完全一致 → 不動
        - 分數沒變(裁判判無效) → 標記為假回合,還原消極計時
        - 分數變了但方向錯(誤判得分方) → 修正分數,維持有效回合
        """
        if not self.rounds:
            return
        
        if info.score_left is None or info.score_right is None:
            return
        
        last = self.rounds[-1]
        ocr_observed = [info.score_left, info.score_right]
        
        # Case 0: 一致 → 不動
        if list(last.score_after) == ocr_observed:
            return
        
        original_after = list(last.score_after)
        
        # 判別是哪種情境
        if list(last.score_before) == ocr_observed:
            # ─── Case B: 假回合 (裁判判無效 / 燈號誤閃) ───
            last.score_after = ocr_observed
            last.score_invalidated = True
            last.is_disputed = True
            last.dispute_type = "void"
            last.details = (
                (last.details + " | " if last.details else "")
                + f"假回合修正: 系統判定 {original_after} 但 OCR 觀察分數無變化({ocr_observed}),"
                f"判定為裁判判無效或燈號誤閃,此回合不應計入有效得分"
            )

            logger.warning(
                f"[幀 {info.frame_number}] 假回合修正: 上回合 {last.filename} "
                f"score_after {original_after} → {ocr_observed},將保留消極錨點"
            )
            self.last_record_needs_rewrite = True
            
        else:
            # ─── Case A: 誤判得分方(或分數差異) ───
            last.score_after = ocr_observed
            last.score_invalidated = True
            last.is_disputed = True
            last.dispute_type = "wrong_winner"
            
            # 推斷實際得分方
            delta_left = ocr_observed[0] - last.score_before[0]
            delta_right = ocr_observed[1] - last.score_before[1]
            if delta_left > 0 and delta_right > 0:
                actual = "雙方"
            elif delta_left > 0:
                actual = "左方"
            elif delta_right > 0:
                actual = "右方"
            else:
                actual = "無人(分數倒退?)"
            
            last.details = (
                (last.details + " | " if last.details else "")
                + f"得分方修正: 系統判定 {last.winner} 得分 → 實際為 {actual} 得分,"
                f"分數 {original_after} → {ocr_observed}"
            )
            logger.warning(
                f"[幀 {info.frame_number}] 得分方修正: "
                f"上回合 {last.filename} 系統判 {last.winner} → 實際 {actual}, "
                f"分數 {original_after} → {ocr_observed}"
            )
            # 注意: 不還原消極計時,因為真的有人得分,就算判錯邊也算一次有效的得分交鋒
        
        self.last_record_needs_rewrite = True

    def get_point(self, info: FrameInfo) -> RoundRecord:

        # 切出該回合影片。記錄亮燈結果與獲勝方（左 / 右 / 雙方）。no_score_added_count 歸零

        sl = self.prev_round_score[0]
        sr = self.prev_round_score[1]

        if info.light_left and info.light_right:
            winner = "both"
            end_type = "double_win"
            sl += 1
            sr += 1
        elif info.light_left:
            winner = "left"
            end_type = "win"
            sl += 1
        elif info.light_right:
            winner = "right"
            end_type = "win"
            sr += 1
        else:
            winner = "none"
            end_type = "none"

        self.last_known_score = [sl, sr]

        record = self.create_round_record(info, sl, sr, winner, end_type)
        self.prev_round_score = [sl, sr]
        #self.no_score_added_count_before_last_round = self.no_score_added_count  # 備份
        self.no_score_added_count = 0.0
        # ★ 新增:得分判定已完成,清除停頓燈號記憶 ★
        # 這次得分已經「用掉」了亮燈事件,旗標的任務結束。
        # 若不清,當 timer 卡死不動時(in_round 第 1073 行的歸零永遠觸發不了),
        # 殘留的 True 會在下一次停頓判定時被當成幽靈燈,再次誤判得分。
        self.light_seen_during_stall_left = False
        self.light_seen_during_stall_right = False
        self.board_changed_seen_during_stall = False

        # 附帶檢查 — 是否為整場比賽結束
        self.check_match_end(record, sl, sr)
         # 新增:0:00 壓哨得分的特殊處理
        if info.timer_value == 0.0:
            self.after_period_end = True
            logger.debug(f"[幀 {info.frame_number}] 偵測到 0:00，開啟局間休息忽略模式")

        logger.info(
            f"[幀 {info.frame_number}] === 回合 {self.round_count} 結束 === "
            f"類型={end_type}, 獲勝={winner} (由燈號判定), 真實推算比分更新為: [{sl}, {sr}]" 
            # 我不要按照影片上的，這裡輸出的最好是實際的分數變化，以供後續作其他事。
        )

        self.finalize_round(record, info)
        return record

    def times_up(self, info: FrameInfo,
                              score_left: Optional[int],
                              score_right: Optional[int]) -> RoundRecord:
       

        sl = score_left if score_left is not None else self.prev_round_score[0]
        sr = score_right if score_right is not None else self.prev_round_score[1]

        record = self.create_round_record(info, sl, sr, "none", "period_end")

        # 附帶檢查 — 是否為整場比賽結束
        self.check_match_end(record, sl, sr)

        logger.info(
            f"[幀 {info.frame_number}] === 回合 {self.round_count} 結束 === "
            f"類型=period_end, 局={self.current_period} 時間到, 比分=[{sl}, {sr}]"
        )

        """
        # 判斷是否進入延長賽或等待下一局
        if not record.match_end:
            if self.current_period == 3 and sl == sr:
                # 「3/3 局結束時雙方平手，且計時器重置為 1:00 附近開始倒數。」
                logger.info("3/3 局結束平手，進入延長賽等待")
            self.state = State.WAITING_FOR_NEW_PERIOD
        
        """
        self.finalize_round(record, info)
        return record

    def end_round_passivity(self, info: FrameInfo,
                             score_left: Optional[int],
                             score_right: Optional[int]) -> RoundRecord:
        sl = score_left if score_left is not None else self.prev_round_score[0]
        sr = score_right if score_right is not None else self.prev_round_score[1]

        self.passivity_count += 1

        # 兩次消極 會拿到紅牌 這樣兩邊都會被加到分
        if self.passivity_count >= 2:
            sl += 1
            sr += 1
            self.last_known_score = [sl, sr]
            logger.info(f"  [消極比賽] 偵測到第 {self.passivity_count} 次消極，雙方各加一分")

        record = self.create_round_record(info, sl, sr, "none", "passivity")
        self.check_match_end(record, sl, sr)

        # no_score_added_count 歸零
        self.no_score_added_count = 0.0

        logger.info(
            f"[幀 {info.frame_number}] === 回合 {self.round_count} 結束 === "
            f"類型= 消極比賽, 比分=[{sl}, {sr}]"
        )

        self.finalize_round(record, info)
        return record

    def create_round_record(self, info: FrameInfo,
                             score_left: int, score_right: int,
                             winner: str, end_type: str) -> RoundRecord:
        filename = f"test{self.round_count:03d}"
        #is_bonus_match = self.current_period == 4

        record = RoundRecord(
            filename=filename,
            period=self.current_period,
            round_number=self.round_count,
            time_startStamp_for_MAMA=self.format_timestamp(self.current_round_start_time),
            time_endStamp_for_MAMA=self.format_timestamp(info.timestamp),
            start_frame=self.current_round_start_frame,
            end_frame=info.frame_number,
            timer_start=self.format_timer(self.current_round_timer_start),
            timer_end=self.format_timer(info.timer_value),
            score_before=list(self.prev_round_score),
            score_after=[score_left, score_right],
            winner=winner,
            end_type=end_type,
            #bonus_match=is_bonus_match,
            #gap_before=self.gap_pending,
        )
        return record

    def finalize_round(self, record: RoundRecord, info: Optional[FrameInfo] = None):
        self.rounds.append(record)
        self.timer_stall_count = 0
        self.timer_stall_start_frame = 0 #反正局都已經

        # 新增:只有「得分類」結束才記錄得分幀號,供 waiting_for_start 擋導播遲滯用(先不用 我後來把暫停時間改成 45 幀了)
        #if record.end_type in ("win", "double_win") and info is not None:
        #   self.last_score_frame = info.frame_number

        # too_late_to_win = float(record.timer_end.replace(":", ".")) # 這個先不要用到，相信裁判不會直接沒剩幾秒就切
        if record.match_end: # 全打完了
            self.state = State.MATCH_ENDED
        elif record.end_type == "period_end" or record.timer_end == "0.00": # 這局結束了 這裡是字串
            # 這裡感覺要注意 不知道 timer_end 是否可以正確變成 0.00 也有可能長得像 0:00, 000 要看 ocr 怎麼辨識。
            self.state = State.WAITING_FOR_NEW_PERIOD
            self.last_known_timer = 180 
            # 原本是 none 現在提前更新給下一局的 開始時間用，這樣parse_time 那裏才不會判斷錯誤
            self.prev_timer = None
        else: # 局內回合結束，準備下回合
            self.state = State.WAITING_FOR_START
            # 新增:把 prev_timer 對齊到最新值,避免下一幀 waiting_for_start 誤觸發 is_continuation
            if info is not None and info.timer_value is not None:
                self.prev_timer = info.timer_value

    def check_match_end(self, record: RoundRecord,
                         score_left: int, score_right: int):
        """
        檢查整場比賽是否結束。

        五種結束條件：
        1. 得分致勝（任一方達 15 分）
        2. 時間到、比分領先（3/3 局結束且分數不同）
        3. 延長賽得分致勝(先不要做)
        4. 延長賽優先權(先不要做)
        """
        # 1. 得分致勝（任一方達 15 分）
        if record.end_type in ("win", "double_win"):
            if score_left >= 15:
                record.match_end = True
                self.state = State.MATCH_ENDED
                record.match_winner = "left"
                logger.info(f"比賽結束：左方達 {score_left} 分（最高得分可能）")
                return
            if score_right >= 15:
                record.match_end = True
                self.state = State.MATCH_ENDED
                record.match_winner = "right"
                logger.info(f"比賽結束：右方達 {score_right} 分（最高得分可能）")
                return

        # 條件 2：時間到、比分領先
        is_period_3_end = (self.current_period == 3 and 
                           (record.end_type == "period_end" or record.timer_end == "0.00"))

        if is_period_3_end:
            if score_left != score_right:
                record.match_end = True
                self.state = State.MATCH_ENDED
                record.match_winner = "left" if score_left > score_right else "right"
                logger.info(
                    f"比賽結束：3/3 時間到，勝利方: {record.match_winner}方 "
                    f"{score_left}:{score_right}（時間到）"
                )
                return
        
            else:
                # 新增:比分相等 → 需要延長賽,但目前不支援
                record.match_end = True
                record.match_winner = "overtime"
                record.details.append(
                    f"第 3 局結束時比分相等 {score_left}:{score_right},按規則需要進入延長賽。"
                    f"目前系統不支援延長賽賽制處理,中止後續分析。 ciallo (∠·ω )⌒★"
                )
                self.state = State.MATCH_ENDED
                logger.warning(
                    f"★★★ 第 3 局結束時比分平手 {score_left}:{score_right},需要延長賽。"
                    f"目前系統不支援延長賽,將中止處理。"
                )
                logger.warning(
                    f"★★★ 建議:手動編輯後續延長賽片段,或等待系統未來支援。 \
                    被你發現偷懶的地方了 ciallo (∠·ω )⌒★ "
                )
                return
       
    def _detect_clip_jump(self, info: FrameInfo) -> Tuple[bool, bool]:
        """
        回傳 (is_real_clip_jump, should_update_timer)
        - is_real_clip_jump: True 表示真的是影片剪輯跳變，要呼叫 handle_clip_jump
        - should_update_timer: False 表示這幀 OCR 壞掉，不要更新 last_known_timer
        """
        # 只在「回合進行中」才偵測剪輯跳變
        # WAITING_FOR_START / WAITING_FOR_NEW_PERIOD 期間 timer 可能正常重置,不適用
        if self.state != State.IN_ROUND:
            return False, True
    
        """偵測影片被剪輯的跳變：UI 持續顯示，但 timer 突然大幅減少 + score/period 有變化"""
        
        if info.timer_value is None or self.last_known_timer is None:
            return False, True

        timer_drop = abs(self.last_known_timer - info.timer_value)

        # 基本條件：時間突然少了超過門檻（不是正常倒數或停頓）
        if timer_drop <= self.thresh.clip_jump_timer_delta:
            return False, True

        # 輔助條件：score 或 period 至少有一個變化（避免 OCR 小抖動誤判）
        score_or_period_changed = False
        if info.score_left is not None and self.last_known_score[0] is not None:
            score_or_period_changed = score_or_period_changed or (info.score_left != self.last_known_score[0])
        if info.score_right is not None and self.last_known_score[1] is not None:
            score_or_period_changed = score_or_period_changed or (info.score_right != self.last_known_score[1])
        if info.period is not None and self.last_seen_period is not None:
            score_or_period_changed = score_or_period_changed or (info.period != self.last_seen_period)

        if not score_or_period_changed:
            logger.debug(f"[幀 {info.frame_number}] timer 跳了 {timer_drop:.1f} 秒，但 score/period 無變化，應該是誤讀，所以忽略")
            return False, False

        logger.warning(f"[幀 {info.frame_number}] ★★★ 偵測到影片剪輯跳變！timer \
                       從 {self.format_timer(self.last_known_timer)} 跳到 {self.format_timer(info.timer_value)}")
        return True, True
    

    # 字卡剛剛不見，現在又出現
    # 這邊記得問 cluade 暫停的事 如果切回來剛好發生停頓，那會如何
    def handle_ui_reappear(self, info: FrameInfo):
        """
        字卡重新出現的處理。
        「情況 A — 分數未變：安全。將 last_known_timer 與當前讀數的
         差值加入 no_score_elapsed，無縫接回正常追蹤。」

        「情況 B — 分數改變：字卡消失期間發生了得分。
         根據分數差值推算得分次數
        """
        logger.debug(f"[幀 {info.frame_number}] UI 字卡重新出現")
        current_score = [info.score_left, info.score_right] # ocr 讀到的比分

        if current_score[0] is None:
            current_score[0] = self.last_known_score[0]
        if current_score[1] is None:
            current_score[1] = self.last_known_score[1]
        
        period_changed = False
        if info.period is not None and self.current_period is not None:
            if info.period != self.current_period:
                period_changed = True

        """ 如果 ui 消失且 局數已經變了 那就趕快先取現在的時間點
        (比如目前 frame 然後趕緊總結回合，再趕快回來去做新回合的作業)"""
        # 更 很好我還是做了
        # 情況 A：分數未變 also 局數也沒變
        if not period_changed and (
                (self.last_known_score[0] is None and self.last_known_score[1] is None) or 
                (current_score[0] == self.last_known_score[0] and current_score[1] == self.last_known_score[1])):
            
            # 安全，加入消極計時
            if self.last_known_timer is not None and info.timer_value is not None:
                gap_time = abs(self.last_known_timer - info.timer_value)
                if gap_time > 0:
                    self.no_score_added_count += gap_time
            logger.debug("  分數未變，無縫接回")
            return None

        # 情況 B：分數變了
        if period_changed:
            logger.warning(f"  字卡消失期間局數從 {self.current_period} 變為 {info.period}，準備跨局結算")
        logger.warning(f"  字卡消失前的分數: {self.last_known_score}，重新出現的分數: {current_score}")

        sl = current_score[0] if current_score[0] is not None else 0
        sr = current_score[1] if current_score[1] is not None else 0
        prev_l = self.last_known_score[0] or 0
        prev_r = self.last_known_score[1] or 0

        original_sl, original_sr = sl, sr
        modified = False

        # 這邊是我準備的防呆，以免說 ocr 又亂讀取
        #注意 這邊的方法是如果讀到的分數不太對，那就強制改回原本的分數，
        # 但是不排除在字卡消失期間有得 >=2 分，所以這邊是在賭 ocr 實力怎麼樣。
        if sl - prev_l >= 5 or sl < prev_l: # 如果現在讀到的分數和上一幀的差太多
            logger.info(f"  [防呆] 左比分異常 ({prev_l} -> {sl})，強制修正回 {prev_l}")
            sl = prev_l
            modified = True
        if sr - prev_r >= 5 or sr < prev_r:
            logger.info(f"  [防呆] 右比分異常 ({prev_r} → {sr})，強制修正回 {prev_r}")
            sr = prev_r
            modified = True
        
        if modified:
            logger.info(f"  [最終採用] 分數已從原始誤讀 {original_sl, original_sr} 修改為 [{sl}, {sr}]")
        else:
            logger.info(f"  [防呆] 分數變化在合理範圍，直接採用: [{sl}, {sr}]")
    
        info.score_left = sl
        info.score_right = sr

        # last_known_score = 現在比分板上應該顯示的分數（smaple_interval 幀更新，）。
        # prev_round_score = 這一回合剛開始時的比分（用來算這一回合的「得分變化」）。
        record = None
        # 強制結算舊回合
        if self.state in (State.IN_ROUND, State.TIMER_STALLED):
            winner = "abnormal"
            end_type = "ui_reappear"
            record = self.create_round_record(info, sl, sr, winner, end_type)
            self.check_match_end(record, sl, sr)
            self.finalize_round(record, info)

        self.last_known_score = [sl, sr]
        self.last_known_timer = info.timer_value
        self.prev_timer = info.timer_value
        #self.prev_round_score = [sl, sr]
        self.no_score_added_count = 0.0

        # 然後才開新回合
        self.begin_round(info)
        self.prev_round_score = [sl, sr]   
        return record

    def handle_clip_jump(self, info: FrameInfo) -> Optional[RoundRecord]:
        """處理影片剪輯跳變：與 handle_ui_reappear 的「情況 B」完全一樣的止損流程"""
        logger.debug(f"[幀 {info.frame_number}] 進入及時止損處理")

        # 1. 取得當前 OCR 讀到的分數（防 None）
        current_score = [info.score_left, info.score_right]
        if current_score[0] is None: current_score[0] = self.last_known_score[0]
        if current_score[1] is None: current_score[1] = self.last_known_score[1]

        sl = current_score[0] if current_score[0] is not None else 0
        sr = current_score[1] if current_score[1] is not None else 0
        prev_l = self.last_known_score[0] or 0
        prev_r = self.last_known_score[1] or 0

        # 2. 防呆修正（OCR 可能亂讀）
        modified = False
        if sl - prev_l >= 5 or sl < prev_l:
            logger.info(f"  [防呆] 左比分異常 ({prev_l} -> {sl})，強制修正回 {prev_l}")
            sl = prev_l
            modified = True
        if sr - prev_r >= 5 or sr < prev_r:
            logger.info(f"  [防呆] 右比分異常 ({prev_r} → {sr})，強制修正回 {prev_r}")
            sr = prev_r
            modified = True

        info.score_left = sl
        info.score_right = sr

        if self.state in (State.IN_ROUND, State.TIMER_STALLED):
            winner = "abnormal"
            end_type = "clip_jump"          # 新增 end_type 方便你之後 debug

            # 4. 強制結束「舊回合」（使用當前幀作為 end）
            record = self.create_round_record(info, sl, sr, winner, end_type)
            self.check_match_end(record, sl, sr)

            logger.info(
                f"[幀 {info.frame_number}] === 因影片剪輯跳變 強制結束舊回合 {self.round_count} === "
                f"end_type={end_type}, winner={winner}, score_after=[{sl}, {sr}]"
            )
            self.finalize_round(record, info)

        # 開啟新回合
        self.last_known_score = [sl, sr]
        self.last_known_timer = info.timer_value
        self.prev_timer = info.timer_value
        #self.prev_round_score = [sl, sr]
        self.no_score_added_count = 0.0 # 對新回合來說 此回合的分數是新回合的開始分數
        self.begin_round(info)
        self.prev_round_score = [sl, sr]   

        # 這邊要注意 如果新回合是消極回合，系統會無法確定，因為我不知道這回合真實的開頭應該是甚麼時候
        logger.debug(f"[幀 {info.frame_number}] 剪輯跳變止損完成，新回合已正確開始")
        return record
    

    def check_implicit_end(self, total_frames: int, fps: float):
        """
        收尾處理。三種影片結束情境:
        1. MATCH_ENDED:過程已偵測到真實結束(15 分 / 第三局時間到並有領先) → 不處理
        2. 回合進行中被截斷:
        - 若在第三局最後幾秒 → 視為比賽真實結束(分數高者勝)
        - 否則僅強制結算,end_type='incomplete',不視為比賽結束
        3. 回合之間結束:
        - 若最後回合為第三局 period_end 且後續 >3 秒無動靜 → 視為比賽真實結束
        - 否則不做事(既有回合已存)
        """
        if self.state == State.MATCH_ENDED:
            return
        
        # ===== 情境 2:回合進行中被截斷 =====
        if self.state in (State.IN_ROUND, State.TIMER_STALLED):
            sl = self.last_known_score[0] or 0
            sr = self.last_known_score[1] or 0
            
            fake_info = FrameInfo()
            fake_info.frame_number = total_frames
            fake_info.timestamp = total_frames / fps
            fake_info.timer_value = self.last_known_timer
            
            # 是不是真實的第三局終了?
            is_period3_near_end = (
                self.current_period == 3
                and self.last_known_timer is not None
                and self.last_known_timer <= 0.2 # 如果錄影到第 0.5 秒 之前被卡掉
                and sl < 15 and sr < 15
            )
            
            if is_period3_near_end:
                record = self.create_round_record(fake_info, sl, sr, "none", "period_end")
                if sl != sr:
                    record.match_end = True
                    record.match_winner = "left" if sl > sr else "right"
                    logger.info(f"影片於第三局終了結束,比賽結束,獲勝:{record.match_winner} ({sl}:{sr})")
                else:
                    logger.info(f"影片於第三局終了但比分平手 ({sl}:{sr}),不判定贏家(需延長賽)")
            else:
                record = self.create_round_record(fake_info, sl, sr, "none", "incomplete")
                logger.info(f"影片在回合進行中(或剛好回合結束，時間表正暫停時)被卡掉，比分 {sl}:{sr}, 強制結算")
            
            self.finalize_round(record, None)
            return
        
        # ===== 情境 3:回合之間結束 =====
        if not self.rounds:
            return
        
        last_round = self.rounds[-1]
        sl, sr = last_round.score_after
        remaining_seconds = (total_frames - last_round.end_frame) / fps
        
        # 真實終局條件:最後回合是第三局時間到 + 後面 >3 秒無動靜 + 沒人 15 分
        is_real_match_end = (
            last_round.period == 3
            and last_round.end_type == "period_end"
            and remaining_seconds > self.thresh.ui_hasnt_show_up_min
            and sl < 15 and sr < 15
        )
        
        if is_real_match_end and sl != sr:
            last_round.match_end = True
            last_round.match_winner = "left" if sl > sr else "right"
            self.state = State.MATCH_ENDED
            logger.info(
                f"隱性結束:最後回合為第三局時間到 ({sl}:{sr}),"
                f"後續 {remaining_seconds:.1f} 秒無動靜,比賽結束,獲勝:{last_round.match_winner}"
            )
        else:
            logger.info("影片在回合之間結束,既有回合已正常輸出,無須補處理")

    @staticmethod
    def format_timer(seconds: Optional[float]) -> str:
        if seconds is None:
            return "?:??"
        
        if seconds < 10:
            # 個位數秒數，計時器顯示為小數格式（如 8.2, 4.35, 0.17）包含 0 秒
            return f"{seconds:.2f}"
        
        m = int(seconds) // 60
        s = int(seconds) % 60
        return f"{m}:{s:02d}"

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """將秒數格式化為 HH:MM:SS.ff 時間碼"""
        h = int(seconds) // 3600
        m = (int(seconds) % 3600) // 60
        s = int(seconds) % 60
        f = int((seconds - int(seconds)) * 100)
        return f"{h:02d}:{m:02d}:{s:02d}.{f:02d}"
    
class VideoClipper:

    def __init__(self, source_path: str, output_dir: str, fps: float):
        self.source_path = source_path
        self.output_dir = output_dir
        self.fps = fps
        os.makedirs(output_dir, exist_ok=True)

    def clip_round(self, record: RoundRecord,
                   pause_segments: Optional[List[Tuple[int, int]]] = None):
        """
        切割單一回合影片。

        若有一般暫停(pause_segments)，會裁切掉暫停期間，進行無縫拼接。
        """
        output_path = os.path.join(self.output_dir, f"{record.filename}.mp4")

        start_sec = record.start_frame / self.fps
        end_sec = record.end_frame / self.fps

        if pause_segments and len(pause_segments) > 0:
            # 有中途暫停 -> 需要分段切割後拼接
            logger.info(f"  發現 {len(pause_segments)} 段暫停，我要去切割這些垃圾片段")
            self._clip_with_pauses(output_path, record, pause_segments)
        else:
            # 無暫停 直接切割
            self._clip_simple(output_path, start_sec, end_sec)

        logger.info(f"  影片已切割: {output_path}")

    def _clip_simple(self, output_path: str, start_sec: float, end_sec: float):
        duration = end_sec - start_sec
        cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "cuda",
            "-ss", f"{start_sec:.3f}",
            "-i", self.source_path,
            "-t", f"{duration:.3f}",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-c:a", "aac",
            "-avoid_negative_ts", "make_zero",
            output_path,
        ]
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,      # 不關心,直接丟掉
                stderr=subprocess.PIPE,          # 保留以印錯誤
                check=True,
                timeout=120,
            )
        except FileNotFoundError:
            logger.error("ffmpeg 未安裝或不在 PATH 中")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg 切割失敗: {e.stderr.decode()[:300]}")
            return False
        except subprocess.TimeoutExpired:
            logger.error(f"ffmpeg 仍 timeout (這次真的是運算問題,不是 PIPE 卡住)")
            return False
        
        return True

    def _clip_with_pauses(self, output_path: str, record: RoundRecord,
                          pause_segments: List[Tuple[int, int]]):
        """
        分段切割並拼接（排除暫停期間）要無縫銜接回原本的回合。
        """
        # 計算有效片段
        active_segments = []
        prev_end = record.start_frame
        for pause_start, pause_end in sorted(pause_segments):
            if pause_start > prev_end:
                active_segments.append((prev_end, pause_start))
            prev_end = pause_end
        if prev_end < record.end_frame:
            active_segments.append((prev_end, record.end_frame))

        if not active_segments:
            logger.warning("  沒有有效片段，跳過切割")
            return

        # 如果只有一段，直接切
        if len(active_segments) == 1:
            start_sec = active_segments[0][0] / self.fps
            end_sec = active_segments[0][1] / self.fps
            self._clip_simple(output_path, start_sec, end_sec)
            return

        # 多段：分別切割後用 ffmpeg concat 拼接
        temp_files = []
        concat_list_path = os.path.join(self.output_dir, "_concat_list.txt")

        for i, (seg_start, seg_end) in enumerate(active_segments):
            temp_path = os.path.join(self.output_dir, f"_temp_seg_{i}.mp4")
            if not self._clip_simple(temp_path, seg_start / self.fps, seg_end / self.fps):
                # 任一段失敗就放棄整個 record,把已切的清掉
                for tf in temp_files:
                    if os.path.exists(tf):
                        os.remove(tf)
                logger.error(f"  回合 {record.filename} 暫停切片失敗,跳過拼接")
                return  
            temp_files.append(temp_path)

        # 寫入 concat 列表
        with open(concat_list_path, "w") as f:
            for tf in temp_files:
                f.write(f"file '{tf}'\n")

        # 拼接
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            output_path,
        ]
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                timeout=120,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg 拼接失敗: {e.stderr.decode(errors='ignore')[:300]}")
        except subprocess.TimeoutExpired:
            logger.error(f"ffmpeg 拼接 timeout: {record.filename}")

        # 清理暫存檔
        for tf in temp_files:
            if os.path.exists(tf):
                os.remove(tf)
        if os.path.exists(concat_list_path):
            os.remove(concat_list_path)

    def save_round_json(self, record: RoundRecord):
        json_path = os.path.join(self.output_dir, f"{record.filename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"  json 已儲存: {json_path}")

class MAIN_PIPELINE:
    def __init__(self, roi_config, threshold_config, video_path=None, 
             sample_interval=30.00, input_dir=str(INPUT_DIR), output_dir=str(OUTPUT_DIR)):
        self.video_path = video_path
        self.roi_config = roi_config
        self.threshold_config = threshold_config
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sample_interval = sample_interval

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"無法開啟影片: {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 建立輸出目錄
        match = Path(self.video_path).stem
        output_match_folder = os.path.join(self.output_dir, match) # 先建立一個資料夾
        os.makedirs(output_match_folder, exist_ok=True)

        # 初始化各元件
        ui_reader = UICardReader(self.roi_config, self.threshold_config, frame_width, frame_height)
        detector = RoundDetector(self.threshold_config, fps)
        clipper = VideoClipper(self.video_path, output_match_folder, fps)

        frame_number = 0
        processed_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % self.sample_interval != 0:
                frame_number += 1
                continue

            print(f"\n{'-'*5} [幀 {frame_number} 處理開始]")
            info = ui_reader.read_frame(frame, frame_number, fps, detector.last_known_timer, 
                                                                last_period=detector.last_seen_period,
                                                                prev_ui_visible=detector.prev_ui_visible)
            record = detector.process_frame(info)

            if record is not None:
                # 切割影片
                clipper.clip_round(record, detector.pause_start_and_end)
                # 儲存回合 JSON
                clipper.save_round_json(record)

            # 新增: 上一個 record 被校正過 → 重寫它的 JSON
            if detector.last_record_needs_rewrite:
                clipper.save_round_json(detector.rounds[-1])
                detector.last_record_needs_rewrite = False
                logger.info(f"  上一回合 JSON 已重寫 (因分數校正)")

            # 如果比賽結束
            if detector.state == State.MATCH_ENDED:
                logger.info("=== 整場比賽結束，處理下一部影片或已完成所有影片 ===")
                break

            frame_number += 1
            processed_frames += 1

        cap.release()

        detector.check_implicit_end(total_frames, fps)
        # 強制結算的回合也要切割和存檔
        for record in detector.rounds:
            json_path = os.path.join(clipper.output_dir, f"{record.filename}.json")
            if not os.path.exists(json_path):  # 只處理還沒存過的
                clipper.clip_round(record)
                clipper.save_round_json(record)
                #self._save_match_summary(detector, self.output_dir, fps)

        logger.info(f"處理完成！共 {len(detector.rounds)} 個回合，輸出至 {self.output_dir}")
        pass

def get_roi(video_path: str):
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
                print(f"  ⚠ [{video_name}] 沒有既有的 ROI 設定可以沿用,請重新選擇")
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
            print(f"  ⚠ 無效輸入,請輸入 y / n / s")
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
    h, w = frame.shape[:2]
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
        for video_path in INPUT_DIR.iterdir():
            if video_path.suffix.lower() in OK_VIDEO_FORMATS:
                output_match_folder = OUTPUT_DIR / video_path.stem
            
                if output_match_folder.exists():
                    print(f"跳過 {video_path.name},輸出資料夾已存在: {output_match_folder.name}")
                    continue
                current_roi, new = get_roi(video_path)
                if current_roi is None:
                    # 使用者選擇跳過此影片
                    print(f"已跳過 {video_path.name}\n")
                    continue

                print(f">>> 正在處理 {video_path.name}")
                pipeline = MAIN_PIPELINE(
                    roi_config=current_roi,
                    threshold_config=thresholds,
                    video_path=video_path,
                    sample_interval=15,
                )
                pipeline.run()

            else:
                print(f" 跳過非影片檔 {video_path.name}")
    else:
        print(" [比賽前處理] 輸入路徑不存在")




            