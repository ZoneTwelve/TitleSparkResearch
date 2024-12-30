import requests
import json

# API configuration
API_BASE = "https://api.unieai.com"
MODEL = "UnieAI/Aqua-mini-cloud"
API_KEY = "your_api_key_here"  # Replace with your actual API key

# Headers for authentication
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def generate_title(category):
    """Generate a title using the Chat Completion API."""
    messages = [
        {"role": "system", "content": "你是一個專業的文章標題生成器，負責生成繁體中文標題。"},
        {"role": "user", "content": f"請生成一個五個字的文章標題，並確保輸出僅使用繁體中文，主題為{category}相關。標題必須放在 <title>$title</title>"}
    ]
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 50,
        "temperature": 0.9
    }
    response = requests.post(f"{API_BASE}/v1/chat/completions", headers=HEADERS, json=payload, verify=False)
    response_data = response.json()

    if response.status_code == 200:
        title = response_data["choices"][0]["message"]["content"].strip()
        return title
    else:
        raise Exception(f"Error generating title: {response_data}")

def generate_article(title):
    """Generate an article using the title as a prompt via the Chat Completion API."""
    messages = [
        {"role": "system", "content": "你是一個專業的文章生成器，負責生成繁體中文文章。"},
        {"role": "user", "content": f"請根據以下標題撰寫一篇約500字的文章，並確保輸出僅使用繁體中文：\n\n標題：{title}"}
    ]
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.7
    }
    response = requests.post(f"{API_BASE}/v1/chat/completions", headers=HEADERS, json=payload, verify=False)
    response_data = response.json()

    if response.status_code == 200:
        article = response_data["choices"][0]["message"]["content"].strip()
        return article
    else:
        raise Exception(f"Error generating article: {response_data}")

def save_to_dataset(data, file_path="dataset.jsonl"):
    """Append data to a JSONL file."""
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_existing_dataset(file_path="dataset.jsonl"):
    """Load existing data from a JSONL file to continue generation."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return sum(1 for _ in file)
    except FileNotFoundError:
        return 0
def deduplicate_array(arr):
    seen = set()
    deduplicated = []
    for item in arr:
        if item not in seen:
            deduplicated.append(item)
            seen.add(item)
    return deduplicated

def main(
        model: str,
        api_base: str,
        api_key: str,
        file: str = "dataset.jsonl"
):
    global MODEL, API_BASE, API_KEY
    MODEL = model
    API_BASE = api_base
    API_KEY = api_key

    categories = deduplicate_array([
        "科技", "教育", "文化", "健康", "旅遊", "藝術", "歷史", "心理學", "環境保護", "經濟", 
        "音樂", "電影", "體育", "哲學", "政治", "建築", "時尚", "美食", "自然科學", "社會學",
        "企業", "經營", "法律", "宗教", "數學", "物理學", "化學", "生物學", "地理學", "天文學"
        "國際關係", "戲劇", "舞蹈", "設計", "傳播學", "新聞學", "農業", "工程學", "醫學", "心理學",
        "數位科技", "資訊科技", "人工智慧", "機器學習", "區塊鏈", "加密貨幣", "網路安全", "雲端運算",
        "人際關係", "家庭關係", "職場關係", "友誼", "愛情", "婚姻", "親子關係", "夫妻關係", "父母關係",
        "ICT", "AI", "ML", "DL", "NLP", "CV", "AR", "VR", "IoT", "5G", "6G", "7G", "8G", "9G",
        "IC 設計", "半導體", "電子電路", "電磁學", "電力電子", "電機工程", "通訊工程", "微波工程", "光電工程",
        "加工", "製造", "品管", "品質管理", "專案管理", "產品管理", "行銷", "銷售", "客服", "客戶關係管理",
        "藥品", "醫療器材", "醫療器械", "醫療科技", "醫療保健", "醫療服務", "醫療照護", "醫療資訊", "醫療管理",
        "保養品", "健身", "運動", "運動器材", "運動服飾", "運動科學", "運動心理學", "運動醫學", "運動管理", "運動營養",
        "品茶", "品酒", "品食", "品味", "品味生活", "品味人生", "品味文化", "品味藝術", "品味時尚", "品味生活",
        "3C", "電腦", "手機", "平板", "筆電", "桌機", "網路", "網頁", "程式", "程式設計", "程式開發",
        "零食", "飲料", "食材", "食譜", "烹飪", "烘焙", "燒烤", "燉煮", "炒炸", "煮炊",
        "日本", "台灣", "中國", "韓國", "泰國", "印度", "歐洲", "美洲", "非洲", "澳洲",
        "學校", "高職", "高中", "高職與高中的差異", "大學", "研究所", "博士班", "碩士班", "學士班", "學位班",
        "教育", "教學", "教育學", "教育心理學", "教育科技", "教育管理", "教育行政", "教育政策", "教育法規",
        "衣著", "古著", "現代著", "流行著", "時尚著", "服飾", "服裝", "服飾設計", "服飾搭配", "服飾品牌",
        "衣服", "鞋子", "帽子", "襪子", "內衣", "外套", "褲子", "裙子", "上衣", "下衣",
        "時尚", "流行", "潮流", "風格", "品味", "設計", "設計師", "品牌", "品牌形象", "品牌策略",
        "交通", "運輸", "運送", "運輸工具", "運輸系統", "運輸管理", "運輸規劃", "運輸政策", "運輸法規",
        "國家", "政府", "政治", "政治制度", "政治體制", "政治體系", "政治文化", "政治經濟", "政治社會", "政治生態",
        "駭客", "駭客攻擊", "駭客防禦", "駭客技術", "駭客工具", "駭客網站", "駭客社群", "駭客組織", "駭客事件", "駭客新聞",
        "寵物", "寵物飼養", "寵物保健", "寵物醫療", "寵物美容", "寵物訓練", "寵物服務", "寵物產品", "寵物用品", "寵物食品",
        "貓咪", "狗狗", "魚兒", "鳥兒", "兔子", "龜兒", "蜥蜴", "蛇", "蜘蛛", "蟲子",
        "冷氣", "冷氣清潔", "冷氣維修", "冷氣安裝", "冷氣保養", "冷氣檢修", "冷氣清洗", "冷氣拆裝", "冷氣維護", "冷氣檢修",
        "暖氣", "暖氣清潔", "暖氣維修", "暖氣安裝", "暖氣保養", "暖氣檢修", "暖氣清洗", "暖氣拆裝", "暖氣維護", "暖氣檢修",
        "餐飲", "餐廳", "餐飲業", "餐飲服務", "餐飲管理", "餐飲設計", "餐飲經營", "餐飲品牌", "餐飲文化", "餐飲產業",
        "工作", "職場", "職業", "職涯", "職場生涯", "職場發展", "職場規劃", "職場管理", "職場溝通", "職場關係",
        "西裝", "洋裝", "禮服", "制服", "校服", "運動服", "休閒服", "時尚服", "時尚品牌", "時尚設計",
        "電影", "電影院", "電影票", "電影場", "電影片", "電影資訊", "電影新聞", "電影網站", "電影社群", "電影專區",
        "音樂", "音樂會", "音樂廳", "音樂學", "音樂家", "音樂劇", "音樂劇場", "音樂劇團", "音樂劇院", "音樂劇團",
        "戲劇", "戲劇院", "戲劇團", "戲劇學", "戲劇家", "戲劇劇", "戲劇劇場", "戲劇劇團", "戲劇劇院", "戲劇劇團",
        "舞蹈", "舞蹈團", "舞蹈團", "舞蹈學", "舞蹈家", "舞蹈劇", "舞蹈劇場", "舞蹈劇團", "舞蹈劇院", "舞蹈劇團",
        "設計", "設計師", "設計學", "設計家", "設計劇", "設計劇場", "設計劇團", "設計劇院", "設計劇團", "設計劇院",
        "新聞", "新聞社", "新聞社群", "新聞社區",
        "新創", "創業", "創新", "創意", "創意產業", "創意設計", "創意行銷", "創意管理", "創意經營", "創意策略",
        "職位 CEO", "職位 CTO", "職位 CFO", "職位 COO", "職位 CMO", "職位 CIO", "職位 CDO", "職位 CRO", "職位 CCO", "職位 CPO",
        "TSMC", "台積電", "聯發科", "鴻海", "台達電", "華碩", "宏碁", "神腦", "緯創", "緯創軟體", "智邦科技", "UnieAI",
        "台灣大學", "清華大學", "交通大學", "成功大學", "政治大學", "中央大學", "中山大學", "中興大學", "中正大學", "中原大學",
        
    ])  # Expanded categories for topic generation
    total_generations = 10000
    dataset_file = file

    # Recover from where it left off
    start_index = load_existing_dataset(dataset_file)
    print(f"Starting from index: {start_index}")

    try:
        for i in range(start_index, total_generations):
            # Choose category based on the current index
            category = categories[i % len(categories)]

            # Step 1: Generate title
            title = generate_title(category)
            print(f"[{i + 1}] ({category}) 生成的標題：{title}")

            # Step 2: Generate article based on the title
            article = generate_article(title)
            print(f"[{i + 1}] 生成的文章完成")

            # Step 3: Save to dataset
            save_to_dataset({"category": category, "title": title, "article": article}, dataset_file)

    except Exception as e:
        print(f"出現錯誤：{e}")

if __name__ == "__main__":
    import fire
    fire.Fire(main)

