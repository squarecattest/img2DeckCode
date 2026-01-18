import cv2
import numpy as np
import os
import sys
from fastapi import FastAPI, WebSocket, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# 1. 自動定位路徑
WEB_DIR = Path(__file__).parent.absolute()      # web/ 資料夾
BASE_DIR = WEB_DIR.parent.absolute()            # 專案根目錄
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"

# 2. 強制加入根目錄到 Python 路徑，確保能 import src.recognizer
sys.path.append(str(BASE_DIR))
try:
    from src.recognizer import CardRecognizer
except ImportError:
    print(f"錯誤：無法載入 src.recognizer。請確保目錄結構正確。路徑：{BASE_DIR}")
    sys.exit(1)

app = FastAPI()

# 3. 掛載資料夾 (使用絕對路徑)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

recognizer = CardRecognizer(confidence_threshold=100)

@app.get("/")
async def get():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(content=f"<h1>Error: 找不到 index.html</h1><p>路徑: {index_path}</p>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

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
                "type": "progress", "current": current, "total": total, "percent": int(current/total*100)
            })

        final_output, grid_img_rel_path = await recognizer.process_image_async(
            image, mode=mode, user_id="web_user", progress_callback=progress_callback
        )

        await websocket.send_json({
            "type": "final", "data": final_output, "image_url": f"/output/{grid_img_rel_path}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    # 這裡的 'web.main:app' 告訴 uvicorn 去 web 資料夾下的 main.py 找 app
    uvicorn.run("web.main:app", host="0.0.0.0", port=8000, reload=True)