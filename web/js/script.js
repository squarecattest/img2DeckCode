// DOM 元素獲取
const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('processBtn');
const previewImg = document.getElementById('previewImg');
const uploadText = document.getElementById('uploadText');
const progressBar = document.getElementById('progressBar');
const statusLabel = document.getElementById('statusLabel');
const countLabel = document.getElementById('countLabel');
const jsonOutput = document.getElementById('jsonOutput');
const langSelect = document.getElementById('langSelect');

// 全域語言暫存
let currentI18n = {};

/**
 * 動態載入語言檔案
 * @param {string} lang - 語言代碼 (zh, ja, en)
 */
async function loadLanguage(lang) {
    try {
        const response = await fetch(`static/lang/${lang}.json`);
        if (!response.ok) throw new Error('Language file not found');
        currentI18n = await response.json();
        
        // 更新 UI 文字
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            if (currentI18n[key]) {
                el.innerText = currentI18n[key];
            }
        });

        // 儲存使用者偏好
        localStorage.setItem('preferredLang', lang);
        console.log(`Language loaded: ${lang}`);
    } catch (error) {
        console.error('Failed to load language:', error);
    }
}

// --- 初始化設定 ---

// 1. 決定初始語言 (偏好 > 瀏覽器語系 > 預設中文)
const defaultLang = localStorage.getItem('preferredLang') || 
                   (navigator.language.startsWith('ja') ? 'ja' : 
                    navigator.language.startsWith('en') ? 'en' : 'zh');

langSelect.value = defaultLang;
loadLanguage(defaultLang);

// 2. 綁定語言切換事件
langSelect.onchange = (e) => loadLanguage(e.target.value);

// --- 圖片處理邏輯 ---

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

// --- WebSocket 辨識邏輯 ---

processBtn.onclick = async () => {
    const file = fileInput.files[0];
    if (!file) return alert(currentI18n.error_file || "Please select a file");

    const mode = document.getElementById('modeSelect').value;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/recognize?mode=${mode}`);
    
    // UI 狀態更新
    processBtn.disabled = true;
    document.getElementById('progressBox').style.display = 'block';

    ws.onopen = async () => {
        const bytes = await file.arrayBuffer();
        ws.send(bytes);
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === "progress") {
            // 更新進度條
            progressBar.style.width = msg.percent + '%';
            countLabel.innerText = `${msg.current} / ${msg.total}`;
            // 動態拼接辨識中的卡名
            statusLabel.innerText = `${currentI18n.status_processing}${msg.card_name || '...'}`;
        } 
        
        else if (msg.type === "final") {
            // 顯示最終辨識圖片
            const fullGridImg = document.getElementById('fullGridImg');
            fullGridImg.src = msg.image_url;
            fullGridImg.style.display = 'block';
            document.getElementById('imageResultCard').style.display = 'block';

            // 格式化 JSON 並調用 Prism 高亮
            jsonOutput.textContent = JSON.stringify(msg.data, null, 4);
            if (typeof Prism !== 'undefined') {
                Prism.highlightElement(jsonOutput);
            }
            
            document.getElementById('jsonResultCard').style.display = 'block';
            statusLabel.innerText = currentI18n.status_done;
            processBtn.disabled = false;
            ws.close();
        }

        else if (msg.type === "error") {
            alert((currentI18n.error_ws || "Error") + ": " + msg.message);
            processBtn.disabled = false;
        }
    };

    ws.onerror = () => {
        alert(currentI18n.error_ws || "Connection Error");
        processBtn.disabled = false;
    };
};