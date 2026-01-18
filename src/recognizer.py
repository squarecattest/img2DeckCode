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
        
        # --- 比例校準：針對 60 卡模式大幅加寬 ax 範圍 ---
        # ay1, ay2 控制高度(避開屬性/星等)；ax1, ax2 控制寬度
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

    def _get_grid_by_template(self):
        self._log(">>> 執行 60 卡模式模板比對 (校準邊緣中...)")
        template_path = self.base_dir / 'data' / '60.png'
        img = cv2.imread(str(template_path))
        if img is None: return [], (1, 1)
        
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([110, 150, 150]), np.array([130, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        temp_grid = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            # 針對長條區域(通常是12張一列)進行切分
            if cw > ch * 4:
                card_w = cw / 12
                for i in range(12):
                    temp_grid.append({'x': int(x + i * card_w), 'y': y, 'w': int(card_w), 'h': ch})
            elif ch > 20:
                temp_grid.append({'x': x, 'y': y, 'w': cw, 'h': ch})
        
        temp_grid.sort(key=lambda b: (b['y'], b['x']))
        return temp_grid, (h, w)

    def _get_grid_by_inference(self, image):
        img_h, img_w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(cv2.medianBlur(gray, 5), 45, 80)
        dilated = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if (img_h * img_w * 0.002) < cv2.contourArea(c) < (img_h * img_w * 0.1):
                x, y, w, h = cv2.boundingRect(c)
                if 1.3 < (h/w) < 1.6:
                    if not any(abs(x-bx)<20 and abs(y-by)<20 for bx,by,bw,bh in boxes): boxes.append([x, y, w, h])
        if not boxes: return []
        u_w, u_h = int(np.median([b[2] for b in boxes])), int(np.median([b[3] for b in boxes]))
        def group(coords, thr):
            coords.sort()
            res = [[coords[0]]]
            for c in coords[1:]:
                if c - res[-1][-1] < thr: res[-1].append(c)
                else: res.append([c])
            return [int(np.mean(g)) for g in res]
        xs, ys = group([b[0] for b in boxes], u_w//2), group([b[1] for b in boxes], u_h//2)
        return [{'x': gx, 'y': gy, 'w': u_w, 'h': u_h} for gy in ys for gx in xs]

    def _extract_art_and_match(self, card_crop, ratios, x_shift_pixel=0):
        rh, rw = card_crop.shape[:2]
        # 加上微調位移
        ay1, ay2 = int(rh*ratios['ay1']), int(rh*ratios['ay2'])
        ax1 = int(rw*ratios['ax1']) + x_shift_pixel
        ax2 = int(rw*ratios['ax2']) + x_shift_pixel
        
        # 防止越界
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
        
        if mode == "TEMPLATE_60":
            grid, (th, tw) = self._get_grid_by_template()
            ratios, sx, sy = self.ratios_60, img_w/tw, img_h/th
        else:
            grid, ratios, sx, sy = self._get_grid_by_inference(image), self.ratios_grid, 1.0, 1.0

        raw_results, canvas = [], image.copy()
        for i, box in enumerate(grid):
            # 1. 座標縮放
            rx, ry, rw, rh = int(box['x']*sx), int(box['y']*sy), int(box['w']*sx), int(box['h']*sy)
            
            # 2. 邊緣補償：如果是 60 卡模式且位於每一列的邊緣，強制向外擴展 5 像素
            if mode == "TEMPLATE_60":
                col = i % 12
                if col == 0: rx -= 5; rw += 5 # 最左
                if col == 11: rw += 5 # 最右
            
            crop = image[max(0,ry):min(img_h,ry+rh), max(0,rx):min(img_w,rx+rw)]
            if crop.size == 0: continue

            # 3. 執行辨識
            match, art = self._extract_art_and_match(crop, ratios, 0)
            
            # 4. 如果信心不足，啟動位移補償搜尋
            if match['p_dist'] > self.retry_p_threshold and match['name'] != "Empty Slot":
                shift_px = int(rw * self.shift_step)
                for s in [-shift_px, shift_px]:
                    m, a = self._extract_art_and_match(crop, ratios, s)
                    if m['p_dist'] < match['p_dist']: match, art = m, a

            if match['name'] == "Empty Slot": continue
            
            self._log(f"Slot {i+1:02d}: {match['name']} | P: {match['p_dist']} | {'✅' if match['p_dist'] < self.threshold and match['name'] != 'Unknown' else '❌'}")
            raw_results.append(match)
            
            color = (0, 255, 0) if match['p_dist'] < 80 else (0, 255, 255) if match['p_dist'] < 95 and match['name'] != 'Unknown' else (0, 0, 255)
            cv2.rectangle(canvas, (rx, ry), (rx+rw, ry+rh), color, 3)
            
            if art is not None:
                cv2.imwrite(str(debug_path / f"Slot_{i+1:02d}_{self._sanitize_filename(match['name'])}_P{match['p_dist']}.jpg"), art)
            
            if progress_callback: await progress_callback(i + 1, len(grid)); await asyncio.sleep(0)

        final_json = self._format_output(raw_results, user_id)
        cv2.imwrite(str(debug_path / "_full_grid.jpg"), canvas)
        with open(debug_path / "deck_result.json", "w", encoding="utf-8") as f: json.dump(final_json, f, indent=4, ensure_ascii=False)
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