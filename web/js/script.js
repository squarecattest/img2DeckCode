const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('processBtn');
const previewImg = document.getElementById('previewImg');
const uploadText = document.getElementById('uploadText');
const progressBar = document.getElementById('progressBar');
const statusLabel = document.getElementById('statusLabel');
const countLabel = document.getElementById('countLabel');
const jsonOutput = document.getElementById('jsonOutput');

fileInput.onchange = (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            previewImg.src = event.target.result;
            previewImg.style.display = 'block';
            uploadText.style.display = 'none';
            processBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
};

processBtn.onclick = async () => {
    const file = fileInput.files[0];
    if (!file) return;

    const mode = document.getElementById('modeSelect').value;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/recognize?mode=${mode}`);
    
    processBtn.disabled = true;
    document.getElementById('progressBox').style.display = 'block';

    ws.onopen = async () => {
        const bytes = await file.arrayBuffer();
        ws.send(bytes);
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === "progress") {
            progressBar.style.width = msg.percent + '%';
            countLabel.innerText = `${msg.current} / ${msg.total}`;
            // 顯示當前卡片名稱
            statusLabel.innerText = `辨識中: ${msg.card_name || '...'}`;
        } 
        
        else if (msg.type === "final") {
            // 顯示辨識圖
            const fullGridImg = document.getElementById('fullGridImg');
            fullGridImg.src = msg.image_url;
            fullGridImg.style.display = 'block';
            document.getElementById('imageResultCard').style.display = 'block';

            // 格式化 JSON 並執行 Prism 高亮
            jsonOutput.textContent = JSON.stringify(msg.data, null, 4);
            Prism.highlightElement(jsonOutput);
            document.getElementById('jsonResultCard').style.display = 'block';

            statusLabel.innerText = "辨識完成！";
            processBtn.disabled = false;
            ws.close();
        }

        else if (msg.type === "error") {
            alert("錯誤: " + msg.message);
            processBtn.disabled = false;
        }
    };
};