import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def generate_title(article, model_path="title_generation_model", max_input_length=512, max_output_length=128, device=None):
    # Determine the device to use
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Load the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    
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

if __name__ == "__main__":
    # Example article
    example_article = (
        #"Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are "
        #"programmed to think like humans and mimic their actions. The term may also be applied to any machine "
        #"that exhibits traits associated with a human mind such as learning and problem-solving."
        "數位時代中，網路已成為人們日常生活中不可或缺的一部分，無論是工作、學習還是休閒娛樂，都離不開網路的支援。然而，網路也存在著安全隱患，駭客為   竊取機密資料或破壞系統，時時威脅著網路世界的安全。\n\n駭客王國網站破解\n\n最近，有一個名為『駭客王國』的網站出現，此網站聲稱提供各種駭客技術和安全解決方案，吸引了眾多網路愛好者和專業人士的關注。然而，有些人懷疑   個網站的真實性，擔心他們的資料會被駭客竊取或濫用。為此，我決定深入調查這個網站，揭開其神秘面紗。\n\n第一步：收集資料\n\n我先從搜尋引擎開始，蒐集與『駭客王國』相關的資料，包括論壇、部落格和新聞報導等。根據所收集   的資訊，我發現『駭客王國』這個網站似乎是由一群專業的駭客組成，他們宣稱擁有多年的駭客經驗，可以提供高效的駭客服務，如入侵系統、資料竊取、網路攻擊等。\n\n第二步：網站分析\n\n我利用網路分析工具，如Wireshark和Nmap   對『駭客王國』網站進行深度分析。結果顯示，該網站採用了先進的加密技術和防火牆設置，以保護用戶隱私和資料安全。雖然這種保護對一般使用者來說是好事，但對於想入侵網站的人來說，卻構成了嚴峻的挑戰。\n\n第三步：測試駭客   術\n\n為了更準確地評估『駭客王國』網站的安全性，我決定親自下海測試他們的駭客技術。我偽裝成客戶，向網站提出入侵公司網路的請求，並要求他們提供相應的服務。經過一番交涉，我成功地與網站負責人取得聯絡，他同意為我提供   侵服務。\n\n第四步：入侵測試\n\n在與網站負責人確認好入侵目標後，我準備開始實際操作。在入侵過程中，我發現『駭客王國』網站的確擁有出色的駭客技術，他們利用各種漏洞和手法順利地入侵了目標網路。然而，整個過程也十分危   ，我曾多次遭遇網路防火牆的攔截和警告。\n\n第五步：結論\n\n經過一系列的測試和分析，我對『駭客王國』網站的安全性有了更清晰的認識。雖然他們的駭客技術高超，但同時也存在著一定的風險。如果客戶不慎洩露機密資料，可能會   公司造成嚴重損失。因此，我強烈建議客戶在選擇駭客服務時，務必謹慎挑選信譽良好、技術先進的機構，以確保資料和系統的安全。\n\n第六步：後續追蹤\n\n在完成調查後，我決定持續關注『駭客王國』網站的動態。我發現他們經常更   駭客技術和安全解決方案，這表明他們是一個不斷進步和發展的團隊。同時，我也提醒網路使用者，在享受網路便利的同時，也要注意保護好自己的資料和系統安全。\n\n結語\n\n隨著科技的不斷進步，駭客技術也日趨複雜。『駭客王國』   站的出現，反映出市場對於駭客服務的需求。然而，如何在確保安全和隱私的前提下使用駭客技術，卻是一個值得深思的問題。我希望我的調查結果能對網路使用者有所幫助，並提醒大家在選擇駭客服務時務必謹慎。網路安全是全體網民的   同責任"
    )

    # Generate title
    generated_title = generate_title(example_article)
    print("=="*10)
    print("Generated Title:", generated_title)
    print("=="*10)

