import os
import json
import subprocess
import logging
from typing import Optional, List, Tuple

from process_frame import RoundRecord   # ← 重要

logger = logging.getLogger(__name__)


def group_rounds_for_clipping(rounds):
    """把『void 假回合 + 後續延續回合』歸成同一組。group[-1] 是收尾(通常有效)那回合。"""
    groups = []
    buffer = []
    for r in rounds:
        buffer.append(r)
        if r.dispute_type != "void":
            groups.append(buffer)
            buffer = []
    if buffer:
        groups.append(buffer)
    return groups

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

    def clip_round_group(self, records):
        """
        一組回合（可能含前面數個 void + 最後一個有效回合）切成一支連貫影片：
        各回合自己的有效區間都保留，回合與回合之間的非回合空檔自動排除，
        各回合內部的 pause_segments 也照樣扣掉。以收尾回合命名。
        """
        if not records:
            return None
        terminer = records[-1]
        output_path = os.path.join(self.output_dir, f"{terminer.filename}.mp4")

        # 普通回合（沒有 void 牽連）→ 走原本單回合流程
        if len(records) == 1:
            self.clip_round(terminer, terminer.pause_segments)
            return output_path

        # 跨回合蒐集有效片段（frame 為單位）；回合之間的討論/非回合空檔自動被排除
        active_segments = []
        for rec in records:
            prev_end = rec.start_frame
            for ps, pe in sorted(rec.pause_segments or []):
                if ps > prev_end:
                    active_segments.append((prev_end, ps))
                prev_end = max(prev_end, pe)
            if prev_end < rec.end_frame:
                active_segments.append((prev_end, rec.end_frame))

        if not active_segments:
            logger.warning("  整組沒有有效片段，跳過切割")
            return None

        # 只有一段直接切
        if len(active_segments) == 1:
            s, e = active_segments[0]
            self._clip_simple(output_path, s / self.fps, e / self.fps)
            return output_path

        # 多段：分別切再 concat
        temp_files = []
        concat_list_path = os.path.join(self.output_dir, "_concat_list.txt")
        for i, (s, e) in enumerate(active_segments):
            temp_path = os.path.join(self.output_dir, f"_temp_seg_{i}.mp4")
            if not self._clip_simple(temp_path, s / self.fps, e / self.fps):
                for tf in temp_files:
                    if os.path.exists(tf):
                        os.remove(tf)
                logger.error(f"  合併切片失敗，跳過拼接: {terminer.filename}")
                return None
            temp_files.append(temp_path)

        with open(concat_list_path, "w") as f:
            for tf in temp_files:
                f.write(f"file '{tf}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            output_path,
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           check=True, timeout=120)
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg 拼接失敗: {e.stderr.decode(errors='ignore')[:300]}")
        except subprocess.TimeoutExpired:
            logger.error(f"ffmpeg 拼接 timeout: {terminer.filename}")

        for tf in temp_files:
            if os.path.exists(tf):
                os.remove(tf)
        if os.path.exists(concat_list_path):
            os.remove(concat_list_path)
        return output_path
    
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


