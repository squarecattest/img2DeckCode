import cv2
import numpy as np
import json
import imagehash
import asyncio
import re
from PIL import Image
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class CardRecognizer:
    try:
        RESAMPLE_METHOD = Image.Resampling.LANCZOS
    except AttributeError:
        RESAMPLE_METHOD = Image.LANCZOS

    def __init__(self, confidence_threshold=100):
        self.threshold = confidence_threshold
        self.base_dir = Path(__file__).parent.parent.absolute()
        self.id_db, self.hash_db = self._load_databases()
        
        self.ratios_60 = {'ay1': 0.22, 'ay2': 0.70, 'ax1': 0.10, 'ax2': 0.90}
        self.ratios_grid = {'ay1': 0.21, 'ay2': 0.71, 'ax1': 0.15, 'ax2': 0.87}
        
        self.retry_p_threshold = 80
        self.shift_step = 0.04 
        self.current_session_path = None

    def _load_databases(self):
        data_folder = self.base_dir / 'data'
        try:
            with open(data_folder / 'id_to_card_data.json', 'r', encoding='utf-8') as f: id_db = json.load(f)
            with open(data_folder / 'card_hashes.json', 'r', encoding='utf-8') as f: hash_db = json.load(f)
            return id_db, hash_db
        except Exception: return {}, {}

    def _log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _sanitize_filename(self, filename):
        s = filename.replace(" ", "_")
        return re.sub(r'(?u)[^-\w.]', '', s)

    def _get_grid_by_inference(self, image):
        img_h, img_w = image.shape[:2]
        
        # --- 影像預處理 ---
        # DEBUG 1: 灰階
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.current_session_path:
            cv2.imwrite(str(self.current_session_path / "debug_1_gray.jpg"), gray)

        # DEBUG 2: 模糊化
        blurred = cv2.medianBlur(gray, 5)
        if self.current_session_path:
            cv2.imwrite(str(self.current_session_path / "debug_2_blurred.jpg"), blurred)

        # DEBUG 3: Canny 邊緣偵測
        edged = cv2.Canny(blurred, 40, 80)
        if self.current_session_path:
            cv2.imwrite(str(self.current_session_path / "debug_3_edged.jpg"), edged)

        # DEBUG 4: 膨脹處理
        dilated = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)), iterations=1)
        if self.current_session_path:
            cv2.imwrite(str(self.current_session_path / "debug_4_dilated.jpg"), dilated)

        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates_img = image.copy()
        raw_boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if (img_h * img_w * 0.002) < area < (img_h * img_w * 0.1):
                x, y, w, h = cv2.boundingRect(c)
                if 1.3 < (h/w) < 1.6:
                    raw_boxes.append({'x': x, 'y': y, 'w': w, 'h': h})
                    cv2.rectangle(candidates_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

        if not raw_boxes:
            self._log("!!! 無法辨識到任何初步候選框")
            return []

        # --- 分排處理 ---
        raw_boxes.sort(key=lambda b: b['y'])
        rows_data = []
        current_row = [raw_boxes[0]]
        for i in range(1, len(raw_boxes)):
            if abs(raw_boxes[i]['y'] - current_row[-1]['y']) < (current_row[-1]['h'] * 0.5):
                current_row.append(raw_boxes[i])
            else:
                rows_data.append(current_row)
                current_row = [raw_boxes[i]]
        rows_data.append(current_row)

        final_grid = []
        row_colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 165, 255)]

        for idx, row in enumerate(rows_data):
            row_y = int(np.mean([b['y'] for b in row]))
            row_h = int(np.median([b['h'] for b in row]))
            row_w = int(np.median([b['w'] for b in row]))
            
            xs = sorted([b['x'] for b in row])
            grouped_xs = [[xs[0]]]
            for x in xs[1:]:
                if x - grouped_xs[-1][-1] < (row_w * 0.5):
                    grouped_xs[-1].append(x)
                else:
                    grouped_xs.append([x])
            
            detected_xs = sorted([int(np.mean(g)) for g in grouped_xs])
            
            # --- 填補頭尾邏輯 ---
            if len(detected_xs) >= 2:
                gaps = [detected_xs[i+1] - detected_xs[i] for i in range(len(detected_xs)-1)]
                avg_gap = int(np.median(gaps))
            else:
                avg_gap = int(row_w * 1.1)

            first_x = detected_xs[0]
            while first_x - avg_gap > (img_w * 0.02):
                first_x -= avg_gap
                detected_xs.insert(0, first_x)
            
            last_x = detected_xs[-1]
            while last_x + avg_gap + row_w < (img_w * 0.98):
                last_x += avg_gap
                detected_xs.append(last_x)

            color = row_colors[idx % len(row_colors)]
            for fx in detected_xs:
                if fx < 0 or fx + row_w > img_w: continue
                final_grid.append({'x': fx, 'y': row_y, 'w': row_w, 'h': row_h})
                cv2.rectangle(candidates_img, (fx, row_y), (fx + row_w, row_y + row_h), color, 2)
                cv2.putText(candidates_img, f"R{idx}", (fx, row_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if self.current_session_path:
            cv2.imwrite(str(self.current_session_path / "debug_5_candidates.jpg"), candidates_img)

        self._log(f"網格推算完成：共偵測到 {len(rows_data)} 排，補齊後總計 {len(final_grid)} 個卡槽")
        return final_grid

    def _extract_art_and_match(self, card_crop, ratios, x_shift_pixel=0):
        rh, rw = card_crop.shape[:2]
        ay1, ay2 = int(rh*ratios['ay1']), int(rh*ratios['ay2'])
        ax1 = int(rw*ratios['ax1']) + x_shift_pixel
        ax2 = int(rw*ratios['ax2']) + x_shift_pixel
        
        ax1, ax2 = max(0, ax1), min(rw, ax2)
        art_roi = card_crop[ay1:ay2, ax1:ax2]
        
        if art_roi.size == 0: 
            return {"name": "Unknown", "type": "Unknown", "p_dist": 999}, None
        
        gray = cv2.cvtColor(art_roi, cv2.COLOR_BGR2GRAY)
        if np.std(gray) < 15: 
            return {"name": "Empty Slot", "type": "None", "p_dist": 999}, None

        pil_img = Image.fromarray(cv2.cvtColor(art_roi, cv2.COLOR_BGR2RGB)).resize((128, 128), self.RESAMPLE_METHOD)
        curr_hash = imagehash.phash(pil_img, hash_size=16)
        best_cand, min_dist = {"name": "Unknown", "type": "Unknown", "p_dist": 999}, 999
        
        for cid, data in self.hash_db.items():
            dist = curr_hash - imagehash.hex_to_hash(data["phash"])
            if dist < min_dist:
                min_dist = dist
                info = self.id_db.get(cid, {"name": "Unknown", "type": "Unknown"})
                best_cand = {"id": cid, "name": info["name"], "type": info["type"], "p_dist": int(dist)}
        return best_cand, art_roi

    async def process_image_async(self, image, mode="GRID", user_id="web_user", progress_callback=None):
        img_h, img_w = image.shape[:2]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        debug_path = self.base_dir / 'output' / user_id / f"session_{timestamp}"
        debug_path.mkdir(parents=True, exist_ok=True)
        
        self.current_session_path = debug_path
        
        if mode == "TEMPLATE_60":
            grid, (th, tw) = self._get_grid_by_template()
            ratios, sx, sy = self.ratios_60, img_w/tw, img_h/th
        else:
            grid, ratios, sx, sy = self._get_grid_by_inference(image), self.ratios_grid, 1.0, 1.0

        raw_results, canvas = [], image.copy()
        
        for i, box in enumerate(grid):
            rx, ry, rw, rh = int(box['x']*sx), int(box['y']*sy), int(box['w']*sx), int(box['h']*sy)
            
            if mode == "TEMPLATE_60":
                col = i % 12
                if col == 0: rx -= 5; rw += 5 
                if col == 11: rw += 5 
            
            crop = image[max(0,ry):min(img_h,ry+rh), max(0,rx):min(img_w,rx+rw)]
            if crop.size == 0: continue

            match, art = self._extract_art_and_match(crop, ratios, 0)
            
            if match['p_dist'] > self.retry_p_threshold and match['name'] != "Empty Slot":
                shift_px = int(rw * self.shift_step)
                for s in [-shift_px, shift_px]:
                    m, a = self._extract_art_and_match(crop, ratios, s)
                    if m['p_dist'] < match['p_dist']: match, art = m, a

            if match['name'] == "Empty Slot": continue
            
            self._log(f"Slot {i+1:02d}: {match['name']} P:{match['p_dist']}")
            raw_results.append(match)
            
            color = (0, 255, 0) if match['p_dist'] < 80 else (0, 255, 255) if match['p_dist'] < 95 and match['name'] != 'Unknown' else (0, 0, 255)
            cv2.rectangle(canvas, (rx, ry), (rx+rw, ry+rh), color, 3)
            
            if art is not None:
                cv2.imwrite(str(debug_path / f"Slot_{i+1:02d}_{self._sanitize_filename(match['name'])}_P{match['p_dist']}.jpg"), art)
            
            if progress_callback: await progress_callback(i + 1, len(grid)); await asyncio.sleep(0)

        final_json = self._format_output(raw_results, user_id)
        cv2.imwrite(str(debug_path / "_full_grid.jpg"), canvas)
        with open(debug_path / "deck_result.json", "w", encoding="utf-8") as f: json.dump(final_json, f, indent=4, ensure_ascii=False)
        
        self.current_session_path = None
        return final_json, f"{user_id}/session_{timestamp}/_full_grid.jpg"

    def _format_output(self, results, user_id):
        output = {
            "user_id": user_id,
            "summary": {"total_matched": 0, "threshold": self.threshold},
            "main_deck": {"Monster": {}, "Spell": {}, "Trap": {}},
            "extra_deck": {}
        }
        for data in results:
            if data["p_dist"] < self.threshold and data["name"] != "Unknown":
                name, ctype = data["name"], data["type"].lower()
                output["summary"]["total_matched"] += 1
                if any(k in ctype for k in ['fusion', 'synchro', 'xyz', 'link']):
                    output["extra_deck"][name] = output["extra_deck"].get(name, 0) + 1
                elif "spell" in ctype:
                    output["main_deck"]["Spell"][name] = output["main_deck"]["Spell"].get(name, 0) + 1
                elif "trap" in ctype:
                    output["main_deck"]["Trap"][name] = output["main_deck"]["Trap"].get(name, 0) + 1
                else:
                    output["main_deck"]["Monster"][name] = output["main_deck"]["Monster"].get(name, 0) + 1
        return output