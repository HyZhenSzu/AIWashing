项目说明：该项目用于 LLM 对企业年报等文本中有关 AI Washing 内容的分析研究。

项目版本：version - 0.3
更新日期：20251021
更新内容：

1. Model 目录：
   (1) loader.py: 通过 langchain 框架读取年报文件并对年报文件进行分块。
   (2) scorer.py: 提示词、构建模型和打分逻辑。
   (3) local_settings.py: 路径、API Key 等设置。
   (4) timer.py: 计时器。

2. Aggregate 目录：
   (1) aggregate_scores.py: 用多种方式聚合每份年报的评分。

===================================================================================

环境配置：

1. zai-sdk
2. pypdf
3. langchain
4. langchain_community
5. numpy
6. pandas
