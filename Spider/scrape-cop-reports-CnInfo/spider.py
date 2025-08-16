from FuncScraper import main

customer_req = {
    "file_type": "A股年报",  # 文件类型
    "root_file_path": "C:\Code\Article\Spider\scrape-cop-reports-CnInfo\YearlyReport",  # 文件目录
    "start_date": "2019-01-01",  # 起始日期。默认为 2000-01-01,
    "end_date": "2023-01-01",  # None,  # 结束日期。默认为今天
    "interval": 1,  # 起始日期和结束日期之间的间隔。
    "reverseInterval": 1,  # 从后向前爬
    "workers": 10,  # 同时爬取的线程数。建议最大不要超过CPU线程数的150%。
    "file_download": 1,  # 1：下载到本地； 0：保存到文件
}

if __name__ == '__main__':
    main(customer_req)