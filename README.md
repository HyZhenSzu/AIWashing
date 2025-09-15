项目说明：该项目用于 LLM 对企业年报等文本中有关 AI Washing 内容的分析研究。

项目版本：version - 0.2
更新日期：20250915
更新内容：

1. 在 Model 目录中新增了数个文件，包括：
   (1) loader.py: 使用 langChain 框架读取年报内容，Map-Reduce 汇总整个年报文件的打分
   (2) scorer_glm3.py: 本地部署 ChatGLM3 对文本内容进行打分
   (3) scorer.py: 通过 API 调用 ChatGLM4-Flash 模型
   (4) 其它用于测试功能的文件
2. 在 Model 目录中新增了 Timer.py 文件用于测算时间。

===================================================================================

项目版本：version - 0.1
更新日期：20250816
更新内容：

1. 在年报爬虫项目中添加了 spider.py 文件;
2. 以及在 constant.py 文件中配置目前爬取的年报类型为"信息传输、软件和信息技术服务业";
3. 运行 spider.py 后已经爬取的年报列表"A 股年报.txt"文件。

研究配置：
爬虫项目：https://github.com/Shih-yenh-suan/scrape-cop-reports-CnInfo.git
作者：Shih-yenh-suan
