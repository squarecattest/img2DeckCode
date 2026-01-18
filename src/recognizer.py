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

    def __init__(self, confidence_threshold=90):
        self.threshold = confidence_threshold
        self.base_dir = Path(__file__).parent.parent.absolute()
        self.id_db, self.hash_db = self._load_databases()
        
        self.ratios_60 = {'ay1': 0.22, 'ay2': 0.70, 'ax1': 0.10, 'ax2': 0.90}
        self.ratios_grid = {'ay1': 0.21, 'ay2': 0.71, 'ax1': 0.15, 'ax2': 0.87}
        
        self.retry_p_threshold = 80
        self.shift_step = 0.04 

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

    def _get_grid_by_inference(self, image, session_path=None):
        img_h, img_w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if session_path: cv2.imwrite(str(session_path / "debug_1_gray.jpg"), gray)
        
        blurred = cv2.medianBlur(gray, 5)
        edged = cv2.Canny(blurred, 40, 80) # 稍微調低門檻以抓取更多邊緣
        dilated = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        raw_boxes = []
        candidates_img = image.copy()
        
        for c in contours:
            area = cv2.contourArea(c)
            if (img_h * img_w * 0.002) < area < (img_h * img_w * 0.1): # 放寬面積限制
                x, y, w, h = cv2.boundingRect(c)
                if 1.2 < (h/w) < 1.6: # 放寬比例限制，避免邊緣卡片被剔除
                    raw_boxes.append({'x': x, 'y': y, 'w': w, 'h': h})
                    cv2.rectangle(candidates_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

        if not raw_boxes: return []

        raw_boxes.sort(key=lambda b: b['y'])
        rows_data = []
        current_row = [raw_boxes[0]]
        for i in range(1, len(raw_boxes)):
            if abs(raw_boxes[i]['y'] - current_row[-1]['y']) < (current_row[-1]['h'] * 0.5):
                current_row.append(raw_boxes[i])
            else:
                rows_data.append(current_row); current_row = [raw_boxes[i]]
        rows_data.append(current_row)

        def get_global_ref(rows):
            all_b = [b for r in rows for b in r]
            if not all_b: return None
            ref_w = int(np.median([b['w'] for b in all_b]))
            ref_h = int(np.median([b['h'] for b in all_b]))
            gaps = []
            for r in rows:
                r_sorted = sorted(r, key=lambda b: b['x'])
                for i in range(len(r_sorted)-1):
                    g = r_sorted[i+1]['x'] - r_sorted[i]['x']
                    if ref_w * 0.9 < g < ref_w * 1.6: gaps.append(g)
            ref_gap = int(np.median(gaps)) if gaps else int(ref_w * 1.05)
            return ref_w, ref_h, ref_gap

        m_ref = get_global_ref(rows_data[:5]) or get_global_ref(rows_data)
        m_w, m_h, m_gap = m_ref

        final_grid = []
        row_colors = [(255, 50, 50), (50, 255, 50), (50, 50, 255), (255, 255, 50), (255, 50, 255)]

        def process_and_fill_full(rows_to_process, ref_w, ref_h, ref_gap, label_prefix, color_start_idx):
            for r_idx, row in enumerate(rows_to_process):
                row_y = int(np.mean([b['y'] for b in row]))
                # 找出這一排所有可能的錨點（較寬鬆）
                anchor_xs = {b['x']: b for b in row}
                det_xs = sorted(list(anchor_xs.keys()))
                if not det_xs: continue 

                # 往左推算到盡頭
                left_most = min(det_xs)
                while left_most - ref_gap > (img_w * 0.02):
                    left_most -= ref_gap
                
                # 從最左邊開始往右掃描整張圖
                curr_x = left_most
                potential_xs = []
                while curr_x + (ref_w * 0.5) < img_w: # 只要中心點還在圖內就繼續
                    potential_xs.append(curr_x)
                    curr_x += ref_gap

                color = row_colors[(color_start_idx + r_idx) % len(row_colors)]
                for fx in potential_xs:
                    # 匹配現有的偵測框
                    matched_anchor = None
                    for ax in det_xs:
                        if abs(fx - ax) < (ref_w * 0.5): # 增加匹配容許度
                            matched_anchor = anchor_xs[ax]
                            break
                    
                    target_box = matched_anchor if matched_anchor else {'x': fx, 'y': row_y, 'w': ref_w, 'h': ref_h}
                    is_inferred = matched_anchor is None
                    
                    # 邊界檢查：確保不超出圖片太誇張
                    tx, ty, tw, th = target_box['x'], target_box['y'], target_box['w'], target_box['h']
                    if tx < 0 or tx > img_w - tw*0.3: continue

                    final_grid.append(target_box)
                    
                    # Debug 繪製
                    thickness = 1 if is_inferred else 2
                    cv2.rectangle(candidates_img, (int(tx), int(ty)), (int(tx+tw), int(ty+th)), color, thickness)
                    if is_inferred:
                        cv2.circle(candidates_img, (int(tx+tw//2), int(ty+th//2)), 4, color, -1)

        # 區分主牌組與額外牌組
        main_deck_rows = rows_data[:5]
        extra_deck_rows = rows_data[5:]

        process_and_fill_full(main_deck_rows, m_w, m_h, m_gap, "M", 0)
        
        if extra_deck_rows:
            e_ref = get_global_ref(extra_deck_rows)
            if e_ref:
                process_and_fill_full(extra_deck_rows, e_ref[0], e_ref[1], e_ref[2], "E", len(main_deck_rows))

        if session_path:
            cv2.imwrite(str(session_path / "debug_5_grid_final.jpg"), candidates_img)
        
        return final_grid

    def _extract_art_and_match(self, card_crop, ratios, x_shift_pixel=0):
        rh, rw = card_crop.shape[:2]
        ay1, ay2 = int(rh*ratios['ay1']), int(rh*ratios['ay2'])
        ax1, ax2 = int(rw*ratios['ax1']) + x_shift_pixel, int(rw*ratios['ax2']) + x_shift_pixel
        ax1, ax2 = max(0, ax1), min(rw, ax2)
        art_roi = card_crop[ay1:ay2, ax1:ax2]
        
        if art_roi.size == 0: return {"name": "Unknown", "type": "Unknown", "p_dist": 999}, None
        gray = cv2.cvtColor(art_roi, cv2.COLOR_BGR2GRAY)
        if np.std(gray) < 12: return {"name": "Empty Slot", "type": "None", "p_dist": 999}, None
        
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

    def process_image_sync(self, image, mode="GRID", user_id="web_user", session_id=None, progress_sync_callback=None, progressor=None):
        img_h, img_w = image.shape[:2]
        session_id = session_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        debug_path = self.base_dir / 'output' / user_id / f"session_{session_id}"
        debug_path.mkdir(parents=True, exist_ok=True)
        
        grid = self._get_grid_by_inference(image, session_path=debug_path)
        ratios = self.ratios_grid
        raw_results, canvas = [], image.copy()

        if progressor:
            progressor.setup(len(grid))

        for i, box in enumerate(grid):
            rx, ry, rw, rh = int(box['x']), int(box['y']), int(box['w']), int(box['h'])
            crop = image[max(0,ry):min(img_h,ry+rh), max(0,rx):min(img_w,rx+rw)]
            if crop.size == 0: continue

            match, art = self._extract_art_and_match(crop, ratios, 0)
            p_initial = match['p_dist']
            
            if match['p_dist'] > self.retry_p_threshold and match['name'] != "Empty Slot":
                best_match, best_art = match, art
                for level in [1, 2, 3]:
                    step_px = int(rw * self.shift_step * level)
                    for direction in [-step_px, step_px]:
                        m_s, a_s = self._extract_art_and_match(crop, ratios, direction)
                        if m_s['p_dist'] < best_match['p_dist']:
                            best_match, best_art = m_s, a_s
                    if best_match['p_dist'] < self.retry_p_threshold: break
                match, art = best_match, best_art

            if match['name'] == "Empty Slot": continue
            
            raw_results.append(match)
            # 繪製辨識框
            color = (0, 255, 0) if match['p_dist'] < 80 else (0, 255, 255) if match['p_dist'] < 95 else (0, 0, 255)
            cv2.rectangle(canvas, (rx, ry), (rx+rw, ry+rh), color, 3)
            cv2.putText(canvas, f"P:{match['p_dist']}", (rx, ry+rh+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            if art is not None:
                safe_name = self._sanitize_filename(match['name'])
                cv2.imwrite(str(debug_path / f"Slot_{i+1:02d}_{safe_name}_P{match['p_dist']}.jpg"), art)
            
            if progress_sync_callback:
                progress_sync_callback(i + 1, len(grid), match['name'])

            if progressor:
                progressor.progress()
        if progressor:
            progressor.finished = True
        final_json = self._format_output(raw_results, user_id)
        cv2.imwrite(str(debug_path / "_full_grid.jpg"), canvas)
        with open(debug_path / "deck_result.json", "w", encoding="utf-8") as f: 
            json.dump(final_json, f, indent=4, ensure_ascii=False)
            
        return final_json, f"{user_id}/session_{session_id}/_full_grid.jpg"

    def _format_output(self, results, user_id):
        output = {"user_id": user_id, "summary": {"total_matched": 0, "threshold": self.threshold},
                  "main_deck": {"Monster": {}, "Spell": {}, "Trap": {}}, "extra_deck": {}}
        for data in results:
            if data["p_dist"] < self.threshold and data["name"] != "Unknown":
                name, ctype = data["name"], data["type"].lower()
                output["summary"]["total_matched"] += 1
                if any(k in ctype for k in ['fusion', 'synchro', 'xyz', 'link']):
                    output["extra_deck"][name] = output["extra_deck"].get(name, 0) + 1
                elif "spell" in ctype: output["main_deck"]["Spell"][name] = output["main_deck"]["Spell"].get(name, 0) + 1
                elif "trap" in ctype: output["main_deck"]["Trap"][name] = output["main_deck"]["Trap"].get(name, 0) + 1
                else: output["main_deck"]["Monster"][name] = output["main_deck"]["Monster"].get(name, 0) + 1
        return output