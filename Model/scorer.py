from zai import ZhipuAiClient
import json

class GLM4FlashJsonScorer:
    def __init__(self, api_key: str, model: str = "glm-4-flash"):
        self.client = ZhipuAiClient(api_key=api_key)
        self.model = model

    def score_chunk(self, chunk: str, debug: bool=False) -> float:
        prompt = (
            "AI Washing 是指公司在市场营销中夸大或炒作 AI 相关内容，常见表现包括：\n"
            "1. 使用大量 AI 相关的流行词（如“智能”、“算法”、“深度学习”等），但缺乏实际技术细节。\n"
            "2. 过度宣传 AI 能力，实际应用场景和技术实现不明确。\n"
            "请用数字（0-9）给下面内容的 AI Washing 程度打分，0 代表最低，9 代表最高。\n"
            "直接输出代表分数的数字。\n"
            f"内容：{chunk}\n评分："
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"user", "content": prompt}],
            response_format={"type":"json_object"},
            do_sample=False,
            temperature=0.1,
            top_p=0.5,
            max_tokens=5
        )

        raw = response.choices[0].message.content.strip()
        if debug:
            print("DEBUG: Raw JSON response content:", repr(raw))

        score = None

        # 尝试各种情况
        try:
            parsed = json.loads(raw)
            # 如果 parsed 是 dict
            if isinstance(parsed, dict):
                # 寻找常见的字段
                if "score" in parsed:
                    score = float(parsed["score"])
                elif "评分" in parsed:
                    score = float(parsed["评分"])
                else:
                    # 尝试取所有值里第一个数值
                    for v in parsed.values():
                        if isinstance(v, (int, float)):
                            score = float(v)
                            break
            # 如果 parsed 是 int / float 类型
            elif isinstance(parsed, (int, float)):
                score = float(parsed)
            else:
                # parsed 是字符串或其他类型
                # 尝试从 raw 中提取数字
                import re
                m = re.search(r"[0-9]", raw)
                if m:
                    score = float(m.group(0))
        except json.JSONDecodeError:
            # 如果 json 解析失败，尝试提取数字
            import re
            m = re.search(r"[0-9]", raw)
            if m:
                score = float(m.group(0))

        if score is None:
            # fallback 默认值，可以设为 0 或中间值 4
            score = 0.0

        if debug:
            print("\n=== DEBUG: Chunk & API Score ===")
            print(f"Chunk Content:\n{chunk[:500]}{'...' if len(chunk) > 500 else ''}")
            print(f"Final Score Parsed: {score}\n")

        return score


