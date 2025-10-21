from zai import ZhipuAiClient
import json

class GLM4FlashJsonScorer:
    def __init__(self, api_key, model = "glm-4-flash"):
        self.client = ZhipuAiClient(api_key=api_key)
        self.model = model

    def score_chunk(self, chunk, debug = False):
        prompt = (
            "请根据以下标准判断一段文本的 **AI Washing 程度**（即是否夸大或炒作人工智能相关内容）：\n"
            "\n"
            "【AI Washing 定义】\n"
            "AI Washing 指公司在市场营销或信息披露中，过度使用 AI 相关词汇（如“人工智能”、“算法”、“大模型”、“智能化”、“深度学习”等），"
            "但未提供足够的实际应用细节或技术支撑，以此夸大其技术实力。\n"
            "\n"
            "【评分标准（1–5分）】\n"
            "1 分：完全没有提及 AI 或仅客观提到与 AI 无关的内容。\n"
            "　示例：“公司2023年营业收入同比增长10%，主要得益于产品结构优化。”\n"
            "\n"
            "2 分：仅简要提及 AI 或使用相关术语，但没有夸张成分。\n"
            "　示例：“公司在图像识别项目中尝试应用人工智能技术进行辅助分析。”\n"
            "\n"
            "3 分：存在一定程度的宣传语气，但仍有部分技术或业务细节支持。\n"
            "　示例：“公司利用AI算法提升数据分析效率，优化了部分生产流程。”\n"
            "\n"
            "4 分：明显存在夸张或模糊的表述，缺乏实际技术细节或验证。\n"
            "　示例：“公司自主研发的AI平台将彻底重塑行业格局，引领智能新时代。”\n"
            "\n"
            "5 分：强烈的AI炒作或营销性语言，完全缺乏事实依据。\n"
            "　示例：“我们是全球最智能的AI企业，所有产品都由大模型全面驱动。”\n"
            "\n"
            "请根据以上标准，仅输出数字（1–5）为下面这段内容的AI Washing程度评分。\n"
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
                m = re.search(r"[1-5]", raw)
                if m:
                    score = float(m.group(0))
        except json.JSONDecodeError:
            # 如果 json 解析失败，尝试提取数字
            import re
            m = re.search(r"[1-5]", raw)
            if m:
                score = float(m.group(0))

        if score is None:
            # fallback 默认值，可以设为 0 或中间值
            score = 0.0

        if debug:
            print("\n=== DEBUG: Chunk & API Score ===")
            print(f"Chunk Content:\n{chunk[:500]}{'...' if len(chunk) > 500 else ''}")
            print(f"Final Score Parsed: {score}\n")

        return score


