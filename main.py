import cv2
import numpy as np
import os
import sys
import asyncio
from functools import partial
from fastapi import FastAPI, WebSocket, Query, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.absolute()      
WEB_DIR = BASE_DIR / "web"      
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"

sys.path.append(str(BASE_DIR))
try:
    from src.recognizer import CardRecognizer
except ImportError:
    print(f"錯誤：無法載入 src.recognizer。路徑：{BASE_DIR}")
    sys.exit(1)

app = FastAPI()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

recognizer = CardRecognizer(confidence_threshold=90)

@app.get("/")
async def get():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(content="<h1>Error: 找不到 index.html</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# --- 1. WebSocket 辨識 (網頁用，支援即時進度) ---
@app.websocket("/ws/recognize")
async def websocket_endpoint(websocket: WebSocket, mode: str = Query("GRID")):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    try:
        image_bytes = await websocket.receive_bytes()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            await websocket.send_json({"type": "error", "message": "影像解碼失敗"})
            return

        # 進度回報：由於核心在線程池跑，需透過 loop.call_soon_threadsafe 回傳到 async
        def progress_sync(current, total, card_name):
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({
                    "type": "progress", "current": current, "total": total,
                    "percent": int(current/total*100), "card_name": card_name
                }), loop
            )

        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 在 Executor 中執行 CPU 密集任務，避免卡住其他連線
        final_output, grid_img_rel_path = await loop.run_in_executor(
            None, 
            partial(recognizer.process_image_sync, image, mode=mode, 
                    user_id="web_user", session_id=session_id, progress_sync_callback=progress_sync)
        )

        await websocket.send_json({
            "type": "final", "data": final_output, "image_url": f"/output/{grid_img_rel_path}"
        })

    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()

# --- 2. HTTP API (Discord Bot 用) ---
@app.post("/api/recognize")
async def api_recognize(mode: str = "GRID", file: UploadFile = File(...)):
    loop = asyncio.get_event_loop()
    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(status_code=400, content={"error": "影像解碼失敗"})

        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 使用 run_in_executor 確保多個 POST 請求可以同時進行
        final_output, grid_img_rel_path = await loop.run_in_executor(
            None,
            partial(recognizer.process_image_sync, image, mode=mode, 
                    user_id="dc_bot", session_id=session_id)
        )

        return {
            "type": "final",
            "data": final_output,
            "image_url": f"/output/{grid_img_rel_path}"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)