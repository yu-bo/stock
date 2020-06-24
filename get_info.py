from concurrent.futures import ProcessPoolExecutor, as_completed

from pandas import DataFrame

import stock_sql
import pandas as pd
import numpy as np
import datetime
import time
import tushare as ts

import jieba
import re
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec,word2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.preprocessing import OneHotEncoder


#path = get_tmpfile("word2vec.model")

print(ts.__version__)


ts.set_token("d1af48f518c17415b1b98b2ce84ab7b1a0025adfdde78e22513b31ec")
pro = ts.pro_api()

d = {
    "ts_code": "TS代码",
     "symbol": "股票代码",
     "name": "股票名称",
     "area": "所在地域",
     "industry": "所属行业",
     "fullname": "股票全称",
     "enname": "英文全称",
     "market": "市场类型(主板/中小板/创业板/科创板)",
     "exchange": "交易所代码",
     "curr_type": "交易货币",
     "list_status": "上市状态:L上市 D退市 P暂停上市",
     "list_date": "上市日期",
     "delist_date": "退市日期",
     "is_hs": "是否沪深港通标的，N否 H沪股通 S深股通",
     }

daily = {
    "ts_code": "股票代码",
    "trade_date": "交易日期",
    "open": "开盘价",
    "high": "最高价",
    "low": "最低价",
    "close": "收盘价",
    "pre_close": "昨收价",
    "change": "涨跌额",
    "pct_chg": "涨跌幅(未复权)",
    "vol": "成交量(手)",
    "amount": "成交额(千元)",
}


def dateRange(beginDate, endDate):
    dates = []
    dt = datetime.datetime.strptime(beginDate, "%Y%m%d")
    date = beginDate[:]
    while date <= endDate:
        dates.append(date)
        dt = dt + datetime.timedelta(1)
        date = dt.strftime("%Y%m%d")
    return dates


class stock_basic:
    """
    股票基本信息
    """
    basic_info = {
        "ts_code": "TS代码",
        "symbol": "股票代码",
        "name": "股票名称",
        "area": "所在地域",
        "industry": "所属行业",
        "fullname": "股票全称",
        "enname": "英文全称",
        "market": "市场类型(主板/中小板/创业板/科创板)",
        "exchange": "交易所代码",
        "curr_type": "交易货币",
        "list_status": "上市状态:L上市 D退市 P暂停上市",
        "list_date": "上市日期",
        "delist_date": "退市日期",
        "is_hs": "是否沪深港通标的，N否 H沪股通 S深股通",
    }

    def __init__(self):
        self.file_path ="data/stock_basic.csv"
        self.record_path= "data/data.csv"

    def getBasicInfo(self):
        # 获取信息
        fields = list(self.basic_info.keys())
        headers = list(self.basic_info.values())
        data: DataFrame = pro.query('stock_basic', exchange='',
                                    list_status='L', fields=fields)
        # print(data.values)
        data.columns = headers
        data.to_csv(self.file_path, header=True, encoding='utf-8', index=False)



    def recordDateInfo(self):
        # 初始化跟新时间
        data = pd.read_csv("data/stock_basic.csv", dtype=object)
        # a 股票的全部信息 ，b 要去掉的几列
        a = list(d.values())
        b = ["TS代码", "股票名称", "上市日期", "退市日期"]
        today = datetime.date.today().strftime("%Y%m%d")

        header = [val for val in a if val not in b]
        print(header)
        data_1 = data.drop(header, axis=1)
        data_1.insert(loc=len(data_1.columns), column="更新时间",
                      value=today, allow_duplicates=False)

        for row in data_1.values:
            row[4] = row[2]

        data_1.to_csv("data/data.csv", header=True, encoding='utf-8', index=False)


    def getExVec(self):
        df = stock_sql.getStockFrame(headers=["symbol", "name", "area", "industry"])
        df.sort_values(by=["symbol"], inplace=True, ignore_index=True)
        df.fillna(" ",inplace=True)
        area_vec = OneHotEncoder(sparse=False).fit_transform(df[["area"]])
        indu_vec = OneHotEncoder(sparse=False).fit_transform(df[["industry"]])
        # data = pd.read_csv("data/stock_basic.csv", dtype=object)
        # aa = OneHotEncoder().fit_transform(data[["所在地域"]])
        data_area= {}
        data_indu={}
        symbols = np.array(df[["symbol"]])
        # areas =np.array(df[["area"]]).reshape(-1)
        # data_t={}
        for i in range(len(symbols)):
            data_area[symbols[i][0]] = area_vec[i]
            data_indu[symbols[i][0]] = indu_vec[i]
        return  data_area, data_indu, area_vec.shape,indu_vec.shape







class stock_daily(object):

    def __init__(self):
        pass

    def updateDailyInfo(self):
        """
        更新所有 stock 的日信息
        :return:
        """
        df = stock_sql.getTradeDateList()
        # 按照symbol 升级
        # df.sort_values(by=["symbol"], inplace=True, ignore_index=True)
        stock_list = np.array(df)
        self. updateDailyList(stock_list)
        # print(stock_list)


    def createDateTable(self):
        """
        创建数据库用来记录 更新的日期信息 ,并进行根据 stockBasic信息进行初始化
        :return:
        """
        stock_sql.createDateTable()
        stock_sql.initDateTable()

    def getDailyInfo(self, symbol, ts_code, start_time:str, end_time:str) -> pd.DataFrame:
        """
        获取stock 日信息
        :param symbol: stock 代码
        :param ts_code:
        :param start_time:  起始日期
        :param end_time:    结束日期
        :return:   daily 信息
        """
        df_list = []
        while start_time != end_time:
            df_t = pro.daily(
                ts_code=ts_code, start_date=start_time, end_date=end_time)
            df_list.append(df_t)
            end_time = df_t.tail(1)["trade_date"].values[0]

        if len(df_list) != 0:
            data = pd.concat(df_list, join="inner")
        else:
            data = None

        return data


    def dailyInfoConcat(self, symbol: int, data:pd.DataFrame):
        """
        将新增的日新增信息与原记录的信息合并保存
        :param symbol: stock 代码
        :param data: 新增数据
        :return: 最新更新日期
        """
        path = stock_sql.getFilePath(symbol)
        df = stock_sql.getStockData(path)
        data.trade_date = data.trade_date.astype("int64")
        # 新数据放在前
        data_n = pd.concat([data, df], join='inner', ignore_index=True)
        # 1.数据去重
        data_n.drop_duplicates(subset=["trade_date"], keep='first', inplace=True)
        data_n.to_csv(path, header=True, encoding='utf-8', index=False)
        return data_n.head(1)["trade_date"].values[0]

    def recordLastDate(self,symbol,data):
        """
        记录最后dailyinfo的最后更新日期
        :param symbol: stock 代码
        :param data: 最后更新日期
        :return:  none
        """
        stock_sql.updateTradeDate(symbol, data)

    def updateDailyList(self, stockList):
        """
        更新stocklist中的stock 日信息
        :param stockList:输入的股票列表包含 symbol name tscode  lastdate
        :return:
        """
        today = datetime.date.today().strftime("%Y%m%d")
        for item in stockList:
            # time.sleep(.3)
            print(item)
            data = self.getDailyInfo(item[0], item[2], item[3], "20200312")
            if data is not None:
                last_d = self.dailyInfoConcat(item[0], data)
                self.recordLastDate(item[0], last_d)

    def updateDailyListMutliple(self,stockList):
        """
        更新stocklist中的stock 日信息 (多线程)
        :param stockList:输入的股票列表包含 symbol name tscode  lastdate
        :return:
        """
        #对列表进行拆分
        stock_split = stock_sql.arr_split( stockList ,500)
        executor = ProcessPoolExecutor(max_workers=8)
        all_task = [executor.submit( self.updateDailyList, list_s) for list_s in stock_split]
        for future in as_completed(all_task) :
            res = future.result()


class stock_company(object):
    """
    股票相关行业信息
    """

    def __init__(self):
        self.file_path = "data/stock_company.csv"
        self.company_info = {
            "ts_code": "股票代码",
            "province": "所在省份",
            "city": "所在城市",
            "main_business": "主营业务",
            "business_scope": "经营范围"
        }

    def getCompanyInfo(self):
        df:DataFrame =pro.stock_company(fields =  list( self.company_info.keys()))
        df.to_csv(self.file_path,header=True, encoding='utf-8', index=False)
        print(df)

    def getCompanyData(self):
        drops = ["ts_code"]
        columns =[x for x in self.company_info.keys() if x not in drops ]
        df :DataFrame = pd.read_csv(self.file_path,index_col=False,na_values="")
        df.sort_values(by=["ts_code"], ascending=True, ignore_index=True, inplace=True)
        df = df[columns]
        return df

    def removePunctuation(self,text):
        text = re.sub(r'[{}]+'.format('!,;:?()[]\n，。（）：；、\''), ' ', text)
        return text.strip().lower()

    def word_cut_test(self):
        sentences = ["吸收公共存款",
                     "2015年我毕业于西安科技大学",
                     "2015年我毕业于西安电子科技大学",
                     "2015年我毕业于西安建筑科技大学",
                     "2015年我毕业于西安交通大学",
                     "2015年我毕业于北京大学"]

        for sentence in sentences:
            # 全模式
            words = jieba.cut(sentence, cut_all=True)
            print("全模式:  %s" % " ".join(words))

            words = jieba.cut(sentence, use_paddle=True)
            print("新词模式:  %s" % " ".join(words))
            # 默认精确模式
            words = jieba.cut(sentence)
            print("精确模式:  %s" % " ".join(words))

            # 搜索模式
            words = jieba.cut_for_search(sentence)
            print("搜索模式:  %s" % " ".join(words))

    def word_cut(self):
        jieba.enable_paddle()
        df =self.getCompanyData()
        df =df[["business_scope"]]
        arr = np.array(df)
        stopwords = self.stopwordslist("nlp/stop_words.txt")
        with open("nlp/cut_words.txt","w+", encoding='utf-8') as fw:
            for i in range(len(arr)):
                list=self.removePunctuation(arr[i][0])
                item = jieba.cut(list,use_paddle=True) # 使用paddle模式
                santi_words =[x for x in item if len(x) > 1 and x not in stopwords]
                fw.writelines(santi_words)
                fw.write("\r\n")
                print(santi_words)

    def stopwordslist(self,filepath):
        stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
        return stopwords

    def word2vec(self):
        sentences = word2vec.PathLineSentences("nlp/cut_words.txt")
        model = Word2Vec(sentences, size=20, window=5, min_count=1, workers=4)
        model.save("nlp/word2vec.model")
        model = Word2Vec.load("nlp/word2vec.model")
        # a= model.train([["吸收公众存款", "吸收公众存款"]], total_examples=1, epochs=1)
        vector = model.wv['新材料']
        a=model.similar_by_vector(vector)
        print(a)
        print(vector)

    def doc2vec(self):
        sentences = word2vec.PathLineSentences("nlp/cut_words.txt")
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
        # for  i ,doc in  enumerate(sentences):
        #     ddd = TaggedDocument(doc,[i])
        #     print(ddd)
        model = Doc2Vec(documents, vector_size=20, window=2, min_count=1, workers=4)
        model.save("nlp/doc2vec.model")
        model = Doc2Vec.load("nlp/doc2vec.model")
        vector = model.infer_vector(["电器开关零部件及附件制造"])
        model.similar_by_vector(vector)
        pass



if __name__ == "__main__":
    # stock_record()
    # stock_data()
    # company= stock_company()
    # company.doc2vec()
    basic  =stock_basic()
    basic.getExVec()
