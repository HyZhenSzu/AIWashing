import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
import numpy

class LLMSCorer:
    def __init__(self, model_name="THUDM/chatglm3-6b-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).half().cuda()
        self.model.eval()
        self.device = next(self.model.parameters()).device

        # 准备评分候选 token IDs 列表（"0"–"9"）
        self.rating_texts = [str(i) for i in range(10)]  # "0"–"9"
        self.rating_token_ids = []
        for t in self.rating_texts:
            token_ids = self.tokenizer.encode(t, add_special_tokens=False)
            # 去掉引号或空格等无效字符
            while len(token_ids) > 0:
                first_str = self.tokenizer.decode([token_ids[0]])
                if first_str.strip("' ") == "":
                    token_ids = token_ids[1:]
                else:
                    break
            if len(token_ids) != 1:
                print(f"[Warning] Candidate '{t}' 不是单 token，将被忽略: {token_ids}")
                continue
            decoded = self.tokenizer.decode(token_ids)
            # print(f"[Init] Candidate '{t}' -> token_id: {token_ids[0]} -> decoded: '{decoded}'")
            self.rating_token_ids.append((int(t), token_ids[0]))

    def score_chunk(self, chunk: str) -> float:
        prompt = (
            "AI Washing是指公司在市场营销中夸大或炒作AI相关内容，常见表现包括：\n"
            "1. 使用大量AI相关的流行词（如智能、算法、深度学习等），但缺乏实际技术细节。\n"
            "2. 过度宣传AI能力，实际应用场景和技术实现不明确。\n"
            "请用数字（0~9）给下面内容的 AI Washing 程度打分，0代表最低，9代表最高。\n"
            "直接输出代表分数的数字。\n"
            "示例：\n"
            "内容：我们公司采用了最先进的AI技术，提升了产品智能化水平。\n"
            "评分：8\n"
            "内容：我们产品的AI模块基于深度学习，具体采用了ResNet结构，训练集为ImageNet。\n"
            "评分：2\n"
            "现在请对下面内容打分，直接输出zero到ten的评分：\n"
            f"{chunk}\n评分："
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
                temperature=0.6,
                top_p=0.5,
                pad_token_id=self.tokenizer.eos_token_id
            )

        logits = outputs.scores[0][0]
        probs = F.softmax(logits, dim=-1)

        weight_sum = 0.0
        score_sum = 0.0
        for score, token_id in self.rating_token_ids:
            p = probs[token_id].item()
            weight_sum += p
            score_sum += score * p
        
        weighted_score = score_sum / weight_sum if weight_sum > 0 else 0.0

        # 🔧 这里新增调试打印：原始 chunk + 最终评分
        print("\n=== DEBUG: Chunk & Weighted Score ===")
        print(f"Chunk Content:\n{chunk[:1000]}{'...' if len(chunk) > 1000 else ''}")  # 仅显示前500字
        print(f"Weighted Score: {weighted_score:.3f}\n")

        return weighted_score

    def score_chunk_debug(self, chunk: str) -> float:
        """
        调试版：打印各候选（'0'~'10'）的概率并返回加权均值分数。
        """
        prompt = (
            "AI Washing是指公司在市场营销中夸大或炒作AI相关内容，常见表现包括：\n"
            "1. 使用大量AI相关的流行词（如智能、算法、深度学习等），但缺乏实际技术细节。\n"
            "2. 过度宣传AI能力，实际应用场景和技术实现不明确。\n"
            "请用数字（0~9）给下面内容的 AI Washing 程度打分，0代表最低，9代表最高。\n"
            "直接输出代表分数的数字。\n"
            "示例：\n"
            "内容：我们公司采用了最先进的AI技术，提升了产品智能化水平。\n"
            "评分：8\n"
            "内容：我们产品的AI模块基于深度学习，具体采用了ResNet结构，训练集为ImageNet。\n"
            "评分：2\n"
            "现在请对下面内容打分，直接输出zero到ten的评分：\n"
            f"{chunk}\n评分："
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
                temperature=0.6,
                top_p=0.5,
                pad_token_id=self.tokenizer.eos_token_id
            )

        logits = outputs.scores[0][0]
        probs = F.softmax(logits, dim=-1)

        # 打印 top-5 候选 token
        topk = torch.topk(probs, 5)
        print("\n=== Top-5 Tokens ===")
        for rank, (idx, p) in enumerate(zip(topk.indices, topk.values)):
            decoded = self.tokenizer.decode([idx])
            print(f"{rank+1:2d}: token_id={idx.item():5d}, prob={p.item():.4f}, decoded='{decoded}'")

        weight_sum = 0.0
        score_sum = 0.0
        for score, token_id in self.rating_token_ids:
            p = probs[token_id].item()
            weight_sum += p
            score_sum += score * p
        
        return score_sum / weight_sum if weight_sum > 0 else 0.0
