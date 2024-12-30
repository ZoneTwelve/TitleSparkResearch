import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def generate_title(article, model_path="title_generation_model", max_input_length=512, max_output_length=128):
    # Load the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    #model.to("cuda")
    
    # Prepare the input
    input_encoding = tokenizer(
        article,
        truncation=True,
        padding="max_length",
        max_length=max_input_length,
        return_tensors="pt"
    )

    input_ids = input_encoding["input_ids"]
    #input_ids.to("cuda")
    attention_mask = input_encoding["attention_mask"]

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
	"NLP四大任務 我們在NLP任務當中，可以大致上分為四種:  第一種是分類任務，在分類中資料通常都是文本與相對的label，這類任務會找到文本之間的關係，並通過softmax(多分類)、sigmoid(二分法)來當作輸出的激勵函數，得到最後的結果。  第二種是自然語言生成(Natural Language Generation)，我們在分類任務中資料都是文本與label，在生成任務中就是把label從數字更換成文本資料而已。使用前面的文本資料，來預測後面的文本資料出現的機率，這任務的輸出通常都只是一種文字的分布機率，而不是一個確定的結果。  第三種則是文本相似度檢測(content similarity detection)，文本相似度檢測其實與分類任務的做法相似，分類是使用embedding的結果並通過激勵函數計算出一個確定的結果，而文本相似度檢測是使用embedding的結果，並通過數學式計算，兩個文字或句子之間的高維空間距離，距離越相近的文字相似度就越高。  第四種是序列標註任務(Sequence Tagging)，這個任務會先輸入的文本資料，將每一個文字都手動增加詞性在裡面，例如我喜歡蘋果就會被標註成 [名詞 我] [動詞 喜歡] [名詞 蘋果]。之後留下文字的詞性當作輸出，來完成標註任務。  為什麼要先介紹NLP四大任務呢?因為今天要介紹的T5模型，它將所有的NLP任務引入到一個統一的架構中，意思就是只需一個模型就能夠完成所有NLP任務。"
    )

    # Generate title
    generated_title = generate_title(example_article)
    print("Generated Title:", generated_title)

