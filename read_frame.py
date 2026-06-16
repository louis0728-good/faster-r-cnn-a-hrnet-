import cv2, os, json, logging, math, numpy as np, re, torch, subprocess
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import field
from dataclasses import dataclass, field
from enum import Enum, auto
from rich.logging import RichHandler
from collections import Counter
import easyocr
    

def format_timer(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?:??"
    
    if seconds < 10:
        # 個位數秒數，計時器顯示為小數格式（如 8.2, 4.35, 0.17）包含 0 秒
        return f"{seconds:.2f}"
    
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


if(torch.cuda.is_available()):
    print(" 正在使用 cuda 喔")
else:
    print(" 懷疑正在使用 cpu ")

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
    # 這邊單位都是 s 
    passivity_seconds: float = 50.00 # 比賽開始幾秒了(判定消極比賽用)
    ui_hasnt_show_up_min: float = 3.0 # 最後一個回合結束後，UI 字卡消失後 x 秒內沒出現，代表比賽結束了
    ui_timer_lag: float = 5.0 # 如果 ui 延遲出現可以容忍幾秒
    votes_rate = 0.5
    limit_ocr_height = 80 # 把 ocr 獨到的東西拉長(如果太矮)

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

def set_window_title(cv2_name, zh_title): # 改中文
    try:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, cv2_name)
        if hwnd:
            ctypes.windll.user32.SetWindowTextW(hwnd, zh_title)
    except Exception:
        pass


class ZoomableROI:
    def __init__(self, window_name, image, zh_title="", initial_scale=0.85):
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
    def __init__(self, roi_config, threshold_config, frame_w: int, frame_h: int):
        self.roi_config = roi_config
        self.threshold_config = threshold_config
        self.w = frame_w
        self.h = frame_h


        if easyocr is not None:
            self.reader = easyocr.Reader(["en"], gpu=True, verbose=False)
        else:
            self.reader = None
            logger.warning("打 pip install easyocr")


    def _crop(self, frame: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
        # 切 roi 區域
        rx, ry, rw, rh = roi
        x1 = int(rx * self.w)
        y1 = int(ry * self.h)
        x2 = int((rx + rw) * self.w)
        y2 = int((ry + rh) * self.h)
        return frame[y1:y2, x1:x2]

    def read_timer(self, frame, last_known_timer: Optional[float] = None):
        crop = self._crop(frame, self.roi_config["timer"])
        text, candidates = self.ocr_text(crop, allowlist="0123456789:.", title="時間") # 錨點 時間點
        timer_value = self.parse_timer_text(text, last_known_timer)

        was_repaired = False
        # 如果正常解析失敗，但 OCR 有讀到文字 → 啟動保守救援
        if timer_value is None:
            # last_timer 就是 last_known_timer
            timer_value = self._repair_timer_if_possible(candidates, last_known_timer)
            if timer_value is not None:
                was_repaired = True
                logger.debug(f"  ├─ [OCR 時間] 修補成功: 候選={candidates} → {timer_value}")
        return timer_value, was_repaired

    def read_score_left(self, frame) -> Optional[int]:
        crop = self._crop(frame, self.roi_config["score_left"])
        text, _ = self.ocr_text(crop, allowlist="0123456789", title="左選手成績")
        return self.parse_score_text(text)

    def read_score_right(self, frame) -> Optional[int]:
        crop = self._crop(frame, self.roi_config["score_right"])
        text, _ = self.ocr_text(crop, allowlist="0123456789", title="右選手成績")
        return self.parse_score_text(text)

    def read_period(self, frame, last_known_period: Optional[int] = None):
        crop = self._crop(frame, self.roi_config["period"])
        text, _ = self.ocr_text(crop, allowlist="1234/", title="局")
        period_value = self.parse_period_text(text)
    
        was_repaired = False
        if period_value is None and last_known_period is not None:
            period_value = last_known_period   # 沿用上一幀
            was_repaired = True
        return period_value, was_repaired

    def _variants(self, crop: np.ndarray) -> List[np.ndarray]:
        # ocr 銳利化用的
        USM_AMOUNT = 0.5    # 銳利化強度;輕微 0.3~0.6,>1.0 會出現白邊/光暈
        USM_SIGMA  = 1.0    # 銳利化半徑;小字 0.8~1.2,太大邊緣會糊一圈
        BIL_SIGMA  = 30     # 雙邊濾波去雜訊強度 20~40
        CLAHE_CLIP = 2.0    # 對比強化 2.0~3.0;放大後雜訊太明顯就往下調
         
        h, w = crop.shape[:2]
        ocr_height = self.threshold_config.limit_ocr_height
        if h < ocr_height:
            scale = ocr_height / h
            crop = cv2.resize(crop, (int(w * scale), ocr_height), 
                            interpolation=cv2.INTER_LANCZOS4)
        
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # (1) 保邊去雜訊:把放大後的壓縮雜訊壓掉,但保留筆畫邊緣。就是把邊邊去掉噪音點
        denoised = cv2.bilateralFilter(gray, d=5, # 濾波器視窗
                                   sigmaColor=BIL_SIGMA, sigmaSpace=BIL_SIGMA) # 顏色和空間差距，差多少算相似?
        
        # (2) CLAHE:處理計分板背景漸層/光暈(放在去雜訊之後,避免放大雜訊)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))
        #                   對比度限制放大程度         直方圖均衡化範圍
        # 自適應直方圖均衡化
        enhanced = clahe.apply(denoised)
        
        # (3) Unsharp Mask = 真正的「銳利化」: 1.5*原圖 - 0.5*模糊
        blur  = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=USM_SIGMA)
        sharp = cv2.addWeighted(enhanced, 1.0 + USM_AMOUNT, blur, -USM_AMOUNT, 0)
        variants = [sharp]
        
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)  # 變體 2:標準 Otsu(灰階變黑白)
        
        # 二值化系列「故意」用未銳利化的 enhanced,
        # 否則 USM 的白邊光暈會被 Otsu 切成黑點/斷筆,反而扣分
        _, otsu_inv = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        variants.append(otsu_inv)  # 變體 3:反向(處理深底淺字 / 淺底深字未知的情況)
        
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 31, 10 # 視窗範圍 和 平均值要扣掉的閾值
        ) # 用高斯加權方式
        variants.append(adaptive)  # 變體 4:自適應(處理區域光線不均)
        
        return variants

    def ocr_text(self, crop: np.ndarray, allowlist: str, title:str) -> Tuple[str, List[str]]:
        if self.reader is None:
            return "", []
        
        variants = self._variants(crop)
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
        

        # 當 OCR 讀到 3.00 / 300 / 3:00 且上一刻 timer 很小時 → 直接當成 180 秒
        # 我把 3.0 刪掉了，因為有些比賽影片會以 3.0 作為 3 秒，而非 0:03。就是那麼機掰
        # 這樣做是為了避免系統誤會以為 倒數 3 秒時把它當作 新局開始。
        # last_known_timer 是「總秒數」的 float 型態儲存，68 秒, 33 秒
        if text in ("3.00", "300", "3:00") or (len(text) == 3 and text.startswith("3") and text[1:].isdigit()):
            if last_known_timer is None or last_known_timer == 180.0:   # 可能是新的局回合，或是剛開始錄影
                logger.debug(f"[parse_timer_text] ocr 讀到'3?00'，可能是新的局回合，或是剛開始錄影，所以強制解析為 180 秒 (3分鐘)")
                return 180.0
            

        if len(text) == 4 and text.isdigit():
            if text[1] in ('3', '2'): # 第二個字元如果是 3 or 2 ，我發現系統可能會誤會把 : 認為是 3 or 2
                minutes = int(text[0])
                seconds = int(text[2:])           # 取後面兩位當秒數
                if 0 <= minutes <= 3 and 0 <= seconds <= 59:
                    total = minutes * 60 + seconds
                    return float(total) if total <= 180 else None
            

            
        # 情況 3(原先放後面，但是我發現應該要再嚴格格式前面，不然跑不到)：小數點被 ocr 吃掉的純數字防呆，例如: 7.1 變成 71
        # 只能是純數字，不能有任何其他東西 20260605
        match = re.search(r"^(\d+)$", text.strip())
        if len(text) == 2 and match:
            num_str = match.group(1)
            num = int(num_str)
            
            if last_known_timer is not None:
                # 假設 OCR 把 7.1 讀成了 71，或是 9.0 讀成 90
                # 如果把數字除以 10 後，離上一幀 (例如 7.2) 差距小於 2 秒，就用他
                if len(num_str) == 2 and abs(last_known_timer - (num / 10.0)) <= 2.0:
                    return num / 10.0
                
        # 嚴格格式檢查
        # 合法 timer 格式只有 "S.SS" / "M.SS" / "M:SS"，如果 OCR 讀到 M.SS 但實際上應該是 M:SS，也要在這裡被改回來。
        # 整體必須能完整匹配,否則視為 OCR 異常
        if not re.fullmatch(r"\d{1}[.:]\d{1,2}", text):
            logger.debug(f"[parse_timer_text] '{text}' 格式畸形,交給「格式嘗試修復功能」")
            return None
    
        # 標準格式 M:S 分:秒，順便過濾那些不可能合理的時間(前面都搞格式 現在要去看時間加起來 是否會超過西洋劍比賽的規定賽制時間)
        match = re.search(r"(\d{1}):(\d{2})", text)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            if seconds > 59 or minutes > 3:          # 秒數非法 / 超過上限
                return None
            total = minutes * 60.0 + seconds
            return total if total <= 180.0 else None  # 擋住 3:30=210 這種

        # 情況 2：OCR 讀出小數點 (可能真的是小數，也可能是冒號被誤認)
        # 秒數小數格式（如 8.2 4.35 0.17）
        match = re.search(r"(\d{1})\.(\d{1,2})", text)
        if match:
            digit1 = int(match.group(1))
            digit2 = int(match.group(2))

            # 這樣 9.1 就是 9.1，不會變成 9.01
            possible_sss = float(match.group(0)) # 當作 秒.毫秒 (0.33)
            # 第一位數是 0, 1, 2, 3 (電腦會讀成例如: 0.33、0.46、3.00)
            # 有兩種可能代表，以 0.33 舉例，搞不好其實是: 0:33 秒 或 真的 0.33 毫秒 
            possible_ms = float(digit1 * 60 + digit2)    # 當作 分:秒 (0.33 -> 真實 0:33 但 ocr 讀錯)

            # 擊劍一局最多 3 分鐘。如果第一位數 >=4 或是 ==3 但是後面的尾數並非0(代表已經超過極限了，可以判定應該不是 分:秒  
            # (例如 4.35, 3.32、8.20)
            # 那可以肯定應該是 秒:毫秒 s:ss
            if digit1 >= 4 or (digit1 ==3 and digit2 != 00):
                return possible_sss if possible_sss < 180.0 else None
            
            result = None
            # 如果有歷史時間，看哪一個數字比較合理 (離上一幀比較近)
            if last_known_timer is not None:
                if abs(last_known_timer - possible_ms) <= abs(last_known_timer - possible_sss):
                    result = possible_ms
                else:
                    result = possible_sss
            else:
                """ 這邊有邏輯上的問題，如果第一幀輸入是 0.17 秒這種東西(代表 1ss7 那就只能吃大便了)，不改是因為這個情況很少觸發"""
                # 剛開局沒有歷史紀錄的防呆機制：
                # 第一幀極大機率是正常的 分:秒 (包含 3:00、2:15、0:45 等等)
                # 擊劍通常在 10 秒以下才會顯示小數點。所以如果後面的數字 >= 10，優先當成分:秒
                if digit1 >= 1 or (digit1 == 0 and digit2 >= 10):
                    result = possible_ms

            return result if (result is not None and result < 180.0) else None

        return None
    
    @staticmethod
    # claude 寫的 但是我腦袋 算力用完了 改天再檢查對不對。
    def _repair_timer_if_possible(candidates: List[str], 
                               last_known_timer: Optional[float]) -> Optional[float]:
        """
        當 parse_timer_text 解不出 timer 時的救援。
        用「上一幀的數字 pair」當 anchor,從候選字串中找最合理的 timer。
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
                # 多加一個 如果最左邊那個是誤讀
                attempts.add((int(digits[1]), int(digits[2:4])))
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
        curr_sl: Optional[int],
        curr_sr: Optional[int],
        last_known_score: List[Optional[int]],
    ) -> Tuple[Optional[int], Optional[int], bool]:
        """
        任一側 OCR 沒讀到 → 沿用上一幀。
        不看 light、不看 timer (因為亮燈時 OCR 本來也不會即時更新,沿用等同預期行為)。
        回傳: (修補後 sl, 修補後 sr, 是否有修補)
        """
        if curr_sl is not None and curr_sr is not None:
            return curr_sl, curr_sr, False   # 兩邊都有,不用修
        
        if last_known_score[0] is None and last_known_score[1] is None:
            return curr_sl, curr_sr, False   # 沒歷史值可抄
        
        new_sl = curr_sl if curr_sl is not None else last_known_score[0]
        new_sr = curr_sr if curr_sr is not None else last_known_score[1]
        repaired = (curr_sl is None or curr_sr is None) and new_sl is not None and new_sr is not None
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
        crop = self._crop(frame, self.roi_config["board_left"])
        return self.is_light_on_left(crop)

    def detect_light_right(self, frame: np.ndarray) -> bool:
        crop = self._crop(frame, self.roi_config["board_right"])
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

        s_min = self.threshold_config.light_sat_min
        v_min = self.threshold_config.bright_min
        sat_bright_mask = cv2.inRange(hsv, (0, s_min, v_min), (179, 255, 255))
        final_mask = cv2.bitwise_and(mask_red, sat_bright_mask)
        red_ratio = cv2.countNonZero(final_mask) / final_mask.size
        return red_ratio > self.threshold_config.rg_color_ratio

    def is_light_on_right(self, crop: np.ndarray) -> bool:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        s_min = self.threshold_config.light_sat_min
        v_min = self.threshold_config.bright_min
        g = self.threshold_config.green_range
        
        mask = cv2.inRange(hsv, (g[0], s_min, v_min), (g[1], 255, 255))
        green_ratio = cv2.countNonZero(mask) / mask.size
        return green_ratio > self.threshold_config.rg_color_ratio
    
    # 取得「計分板背景色」
    def get_board_color_left(self, frame: np.ndarray) -> Tuple[float, float, float]:
        crop = self._crop(frame, self.roi_config["board_left"])
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return tuple(np.mean(hsv.reshape(-1, 3), axis=0))

    def get_board_color_right(self, frame: np.ndarray) -> Tuple[float, float, float]:
        crop = self._crop(frame, self.roi_config["board_right"])
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return tuple(np.mean(hsv.reshape(-1, 3), axis=0))
    
    def read_frame(self, frame: np.ndarray, frame_number: int, 
                            fps: float, last_known_timer: Optional[float] = None, last_known_score=[None, None], 
                                                last_known_period=None, prev_ui_visible=True) -> FrameInfo:
        info = FrameInfo()
        info.frame_number = frame_number
        info.timestamp = frame_number / fps if fps > 0 else 0.0

        # 各欄位讀取(timer/period 現在回傳 tuple)
        info.timer_value, timer_repaired = self.read_timer(frame, last_known_timer)
        info.score_left = self.read_score_left(frame)
        info.score_right = self.read_score_right(frame)

        curr_sl = info.score_left
        curr_sr = info.score_right

         # 新增: 沒讀到的分數沿用上一幀
        info.score_left, info.score_right, score_repaired = self._repair_score_if_possible(
            info.score_left, info.score_right, last_known_score or [None, None]
        )
        if score_repaired:
            repaired_msg = []
            if curr_sl is None:
                repaired_msg.append(f"ocr 失效，左側分數直接根據前幀推論 (None → {info.score_left})")
            if curr_sr is None:
                repaired_msg.append(f"ocr 失效，右側分數直接根據前幀推論 (None → {info.score_right})")
            
            logger.debug(
                f"[幀 {frame_number}] [score 修補] {' + '.join(repaired_msg)}"
            )

        info.period, period_repaired = self.read_period(frame, last_known_period)

        # 補償計數投票，看此幀用多少東西去補償
        # 分開計算「左右分數」各算一票,所以總票數最多 4 票(timer/score_left/score_right/period)
        repair_votes = 0
        if timer_repaired:
            repair_votes += 1
        if period_repaired:
            repair_votes += 1
        if curr_sl is None and info.score_left is not None:   # 左分數是補出來的
            repair_votes += 1
        if curr_sr is None and info.score_right is not None:  # 右分數是補出來的
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
            logger.debug(f"[幀 {frame_number}] 嘗試補償後仍缺 {' + '.join(missing)}，判定 UI 消失")
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
                f"  └─ [讀取總結] timer={format_timer(info.timer_value)}, "
                f"score=[{info.score_left}, {info.score_right}], "
                f"period={info.period}, light=[{info.light_left}, {info.light_right}]"
            )
        
        return info