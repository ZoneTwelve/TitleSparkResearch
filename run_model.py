import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
import statistics
model = None
tokenizer = None

def generate_title(article, model_path="title_generation_model", max_input_length=512, max_output_length=128, device=None):
    # Prepare the input
    input_encoding = tokenizer(
        article,
        truncation=True,
        padding="max_length",
        max_length=max_input_length,
        return_tensors="pt"
    )

    input_ids = input_encoding["input_ids"].to(device)
    attention_mask = input_encoding["attention_mask"].to(device)

    # Generate the title
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_output_length,
            num_beams=4,  # Beam search for better results
            early_stopping=True
        )

    # Decode the generated title
    title = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return title

def my_function():
    # Example article
    example_article = (
        #"Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are "
        #"programmed to think like humans and mimic their actions. The term may also be applied to any machine "
        #"that exhibits traits associated with a human mind such as learning and problem-solving."
        #"數位時代中，網路已成為人們日常生活中不可或缺的一部分，無論是工作、學習還是休閒娛樂，都離不開網路的支援。然而，網路也存在著安全隱患，駭客為   竊取機密資料或破壞系統，時時威脅著網路世界的安全。\n\n駭客王國網站破解\n\n最近，有一個名為『駭客王國』的網站出現，此網站聲稱提供各種駭客技術和安全解決方案，吸引了眾多網路愛好者和專業人士的關注。然而，有些人懷疑   個網站的真實性，擔心他們的資料會被駭客竊取或濫用。為此，我決定深入調查這個網站，揭開其神秘面紗。\n\n第一步：收集資料\n\n我先從搜尋引擎開始，蒐集與『駭客王國』相關的資料，包括論壇、部落格和新聞報導等。根據所收集   的資訊，我發現『駭客王國』這個網站似乎是由一群專業的駭客組成，他們宣稱擁有多年的駭客經驗，可以提供高效的駭客服務，如入侵系統、資料竊取、網路攻擊等。\n\n第二步：網站分析\n\n我利用網路分析工具，如Wireshark和Nmap   對『駭客王國』網站進行深度分析。結果顯示，該網站採用了先進的加密技術和防火牆設置，以保護用戶隱私和資料安全。雖然這種保護對一般使用者來說是好事，但對於想入侵網站的人來說，卻構成了嚴峻的挑戰。\n\n第三步：測試駭客   術\n\n為了更準確地評估『駭客王國』網站的安全性，我決定親自下海測試他們的駭客技術。我偽裝成客戶，向網站提出入侵公司網路的請求，並要求他們提供相應的服務。經過一番交涉，我成功地與網站負責人取得聯絡，他同意為我提供   侵服務。\n\n第四步：入侵測試\n\n在與網站負責人確認好入侵目標後，我準備開始實際操作。在入侵過程中，我發現『駭客王國』網站的確擁有出色的駭客技術，他們利用各種漏洞和手法順利地入侵了目標網路。然而，整個過程也十分危   ，我曾多次遭遇網路防火牆的攔截和警告。\n\n第五步：結論\n\n經過一系列的測試和分析，我對『駭客王國』網站的安全性有了更清晰的認識。雖然他們的駭客技術高超，但同時也存在著一定的風險。如果客戶不慎洩露機密資料，可能會   公司造成嚴重損失。因此，我強烈建議客戶在選擇駭客服務時，務必謹慎挑選信譽良好、技術先進的機構，以確保資料和系統的安全。\n\n第六步：後續追蹤\n\n在完成調查後，我決定持續關注『駭客王國』網站的動態。我發現他們經常更   駭客技術和安全解決方案，這表明他們是一個不斷進步和發展的團隊。同時，我也提醒網路使用者，在享受網路便利的同時，也要注意保護好自己的資料和系統安全。\n\n結語\n\n隨著科技的不斷進步，駭客技術也日趨複雜。『駭客王國』   站的出現，反映出市場對於駭客服務的需求。然而，如何在確保安全和隱私的前提下使用駭客技術，卻是一個值得深思的問題。我希望我的調查結果能對網路使用者有所幫助，並提醒大家在選擇駭客服務時務必謹慎。網路安全是全體網民的   同責任"
	"很多飼主在考慮養第二隻或更多貓咪時，常會猶豫要養公貓、還是母貓。日本寵物網《ねこちゃんホンポ》日前分享貓咪三種性別組合的相處狀況，幫助大家做出更明智的選擇。   1.母貓與母貓 兩隻母貓的組合較容易和平相處。由於母貓的地盤意識較弱，性格也較溫和，比較不會把對方視為競爭對手，因此更容易培養出親密的關係。不過，這並不代表母貓之間就完全不會有摩擦，當牠們的個性和年齡差異較大時，衝突的機會就會增加。因此，挑選年齡相近且個性相似的母貓，能夠更順利地建立友好關係。   2.公貓與公貓 公貓具有較強的地盤意識，特別是未結紮的公貓，很容易為了爭奪地盤或是吸引母貓的注意而發生衝突，甚至會在家中出現標記地盤的行為。因此，飼養兩隻公貓時，必須讓每隻公貓擁有各自的休息區、進食區和貓砂盆。更重要的是，在兩隻公貓初次見面時，一定要給予充分的時間讓牠們慢慢適應彼此。   3.公貓與母貓 公貓和母貓的組合被認為是最理想的搭配。公貓通常會對母貓展現出溫柔和保護的態度，而母貓則較容易接受公貓的親近。不過，未結紮的公貓可能會對母貓展現過度的佔有慾，甚至出現攻擊行為。若想要讓牠們和平相處，建議飼養前先替公貓進行絕育，這樣能有效減少攻擊行為，讓牠們更和諧相處。  選擇合適的貓咪性別組合，是打造和諧多貓家庭的第一步。然而，性別只是影響貓咪相處的其中一個因素，飼主還應該留意每隻貓咪的個性特質、生活習慣等，並提供足夠的時間和空間，讓牠們慢慢培養感情，相信很快就能看到貓咪們和樂相處的美好畫面！"
	#"▲柯文哲否認和沈慶京約定行、收賄。（圖／資料畫面）  民眾黨主席柯文哲被控多次和威京集團主席沈慶京密會，地點包括市長辦公室，2人且達成行賄、收賄約定。台北地方法院召開接押庭，柯文哲未否認有密會，反控根本是「臆測」，只有2個人在談，外面怎麼知道談了什麼？質疑檢方為何不去找市長室秘書來問。柯文哲也否認和沈慶京談到京華城的事「他都在說些五四三」。  根據北檢起訴，沈慶京曾透過鼎越開發前董事長朱亞虎，協助安排與柯文哲會面。2020年2月20日，沈慶京和朱亞虎一同前往台北市政府與柯文哲「單獨」會談。會談中，沈慶京得到柯文哲應允，雙方並達成期約賄賂等犯意聯絡。1小時後沈慶京「臉露滿意微笑」走出市長室，門外等候的朱亞虎便陪同一起離開。  是否真有密會並達成行賄、收賄約定？柯文哲回答法官強調「這是典型臆測」。表示只有2個人在裡面，別人怎會知道談什麼？訪客來市長室都有登記紀錄，有會談一定會有登記，應該都是秘書在保管登記表。  法官因此追問，確定現場只有2個人？柯文哲表示，他沒有印象。但也強調，絕對沒有談到容積率，印象中當天沈慶京都在跟他說些「五四三」，很多大老闆都會這樣。"
	#"醫療保健領域，科技的進步帶來了革命性的變化，尤其是智慧醫療的崛起。這項結合了先進科技與醫療保健的領域，正以驚人的速度改變我們診斷、治療、以及管理疾病的方式。本文將探討智慧醫療的未來，以及它如何幫助我們打造更健康、更永續的醫療體系。\n\n一、大數據與人工智慧在醫療保健的應用\n\n大數據與人工智慧(AI)是驅動智慧醫療發展的兩大關鍵技術。大數據讓我們能處理大量、複雜的資料，而AI則讓我們從這些資料中萃取有價值的見解。這兩者的結合，讓醫師能更準確地診斷疾病、預測病患的健康風險，以及制定更有效的治療計畫。\n\n二、遠距醫療與行動醫療\n\n遠距醫療是另一項智慧醫療的重要面向，它讓醫師能透過線上平台，為病患提供醫療諮詢與照護。這在偏鄉或醫療資源不足的地區，尤其有益處。行動醫療則是運用行動裝置與應用程式，協助病患管理他們的健康，並與醫護人員保持聯繫。\n\n三、穿戴式裝置與物聯網\n\n穿戴式裝置如智慧手錶、心率監測器等，已成為監測病患健康狀況的常見工具。這些裝置所蒐集到的資料，可透過物聯網(IoT)傳輸至雲端，讓醫護人員能即時監測病患的狀況，並根據需要提供介入措施。"
    )

    # Generate title
    generated_title = generate_title(example_article, device=device)
    print("=="*10)
    print("Generated Title:", generated_title)
    print("=="*10)

if __name__ == "__main__":
    # Determine the device to use
    device = None
    model_path = "./TitleSpark-v0.2"
    #model_path = "./title_generation_model"
    #model_path = "./title_generation_model_v1.0"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Run on device", device)
    # Load the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

    num_runs = 10
    runtimes = []

    start_time_total = time.perf_counter() #start total timing

    for _ in range(num_runs):  # Single for loop as requested
        start_time = time.perf_counter()
        my_function()  # Call your function
        end_time = time.perf_counter()
        runtimes.append(end_time - start_time) # Collect runtime data
    end_time_total = time.perf_counter() #end total timing

    average_runtime = statistics.mean(runtimes)
    total_runtime = end_time_total - start_time_total

    print(f"Function ran {num_runs} times.")
    print(f"Average runtime: {average_runtime:.6f} seconds")
    print(f"Total runtime for measurements: {total_runtime:.6f} seconds")
    print(f"Minimum runtime: {min(runtimes):.6f} seconds")
    print(f"Maximum runtime: {max(runtimes):.6f} seconds")
    print(f"Standard deviation of runtimes: {statistics.stdev(runtimes):.6f}")

