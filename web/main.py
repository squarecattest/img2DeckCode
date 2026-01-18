import cv2
import numpy as np
import os
import sys
import asyncio
from fastapi import FastAPI, WebSocket, Query, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

WEB_DIR = Path(__file__).parent.absolute()      
BASE_DIR = WEB_DIR.parent.absolute()            
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"

sys.path.append(str(BASE_DIR))
try:
    from src.recognizer import CardRecognizer
except ImportError:
    print(f"錯誤：無法載入 src.recognizer。請確保目錄結構正確。路徑：{BASE_DIR}")
    sys.exit(1)

app = FastAPI()

# 確保輸出目錄存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 掛載靜態檔案
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# 初始化辨識核心
recognizer = CardRecognizer(confidence_threshold=100)

@app.get("/")
async def get():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(content=f"<h1>Error: 找不到 index.html</h1><p>路徑: {index_path}</p>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# --- 1. 原有的 WebSocket 辨識 (供網頁使用) ---
@app.websocket("/ws/recognize")
async def websocket_endpoint(websocket: WebSocket, mode: str = Query("GRID")):
    await websocket.accept()
    try:
        image_bytes = await websocket.receive_bytes()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            await websocket.send_json({"type": "error", "message": "影像解碼失敗"})
            return

        async def progress_callback(current, total):
            await websocket.send_json({
                "type": "progress", 
                "current": current, 
                "total": total, 
                "percent": int(current/total*100)
            })

        # 呼叫核心辨識邏輯
        final_output, grid_img_rel_path = await recognizer.process_image_async(
            image, mode=mode, user_id="web_user", progress_callback=progress_callback
        )

        await websocket.send_json({
            "type": "final", 
            "data": final_output, 
            "image_url": f"/output/{grid_img_rel_path}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()

@app.post("/api/recognize")
async def api_recognize(mode: str = "GRID", file: UploadFile = File(...)):
    try:
        # 讀取上傳圖片
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(status_code=400, content={"error": "影像解碼失敗"})

        final_output, grid_img_rel_path = await recognizer.process_image_async(
            image, mode=mode, user_id="dc_bot", progress_callback=None
        )

        return {
            "type": "final",
            "data": final_output,
            "image_url": f"/output/{grid_img_rel_path}"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    # 注意：reload=True 在開發時很有用，但如果是正式環境請關閉
    uvicorn.run("web.main:app", host="0.0.0.0", port=8000, reload=True)