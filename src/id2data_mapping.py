import json
import requests
import os

def generate_id_to_data_map():
    """
    從 YGOPRODeck API 抓取資料，並建立映射表。
    確保所有異圖 (Alternate Arts) 的 ID 都能對應到正確的卡名與類型。
    """
    url = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_folder = os.path.join(project_root, 'data')
    output_file = os.path.join(data_folder, 'id_to_card_data.json')
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)

    try:
        print(f"正在從 {url} 獲取資料...")
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        raw_data = response.json()
        cards = raw_data.get('data', [])
        
        if not cards:
            print("API 回傳資料為空。")
            return

        id_to_data = {}
        processed_count = 0

        for card in cards:
            card_name = card.get('name', '').replace('"', '')
            card_type = card.get('type', 'Unknown')
            
            card_info = {
                "name": card_name,
                "type": card_type
            }
            
            # --- 重點優化：處理所有關聯的 ID ---
            # 1. 處理主 ID
            main_id = str(card.get('id'))
            id_to_data[main_id] = card_info
            
            # 2. 處理異圖 (card_images 包含所有該卡的圖片 ID)
            image_variants = card.get('card_images', [])
            for img_obj in image_variants:
                img_id = str(img_obj.get('id'))
                if img_id not in id_to_data:
                    id_to_data[img_id] = card_info
            
            processed_count += 1
            if processed_count % 1000 == 0:
                print(f"已處理 {processed_count} 種卡片...")

        # 寫入 JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(id_to_data, f, ensure_ascii=False, indent=4)
        
        print("-" * 30)
        print(f"成功！")
        print(f"總卡片種類：{processed_count}")
        print(f"總 ID 映射數 (含異圖)：{len(id_to_data)}")
        print(f"檔案儲存於：{output_file}")

    except Exception as e:
        print(f"發生錯誤：{e}")

if __name__ == "__main__":
    generate_id_to_data_map()