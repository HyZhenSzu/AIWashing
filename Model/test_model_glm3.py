import torch
from transformers import AutoTokenizer, AutoModel

# print("CUDA available?", torch.cuda.is_available())
# print("Current CUDA version used by PyTorch:", torch.version.cuda)
# print("Compute Device Count:", torch.cuda.device_count())
# if torch.cuda.is_available():
#     print("Device Name:", torch.cuda.get_device_name(0))

def test_chatglm3_6b_base():
    try:
        tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b-base", trust_remote_code=True)
        model = AutoModel.from_pretrained("THUDM/chatglm3-6b-base", trust_remote_code=True).half().cuda()
        model = model.eval()
        
        prompt = "请帮我将以下句子翻译成英文：今天天气很好。"
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=50)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("模型输出：", result)
        print("✅ ChatGLM3-6B-Base 本地运行正常！")
    except Exception as e:
        print("⚠️ 模型加载或推理失败:", e)

if __name__ == "__main__":
    test_chatglm3_6b_base()
