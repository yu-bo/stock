import tushare as ts
print(ts.__version__)

import time
import datetime
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed

import  stock_sql

ts.set_token("d1af48f518c17415b1b98b2ce84ab7b1a0025adfdde78e22513b31ec")
pro = ts.pro_api()

d={"ts_code":"TS代码",
"symbol":"股票代码",
"name":"股票名称",
"area":"所在地域",
"industry":"所属行业",
"fullname":"股票全称",
"enname":"英文全称",
"market":"市场类型(主板/中小板/创业板/科创板)",
"exchange":"交易所代码",
"curr_type":"交易货币",
"list_status":"上市状态:L上市 D退市 P暂停上市",
"list_date":"上市日期",
"delist_date":"退市日期",
"is_hs":"是否沪深港通标的，N否 H沪股通 S深股通",
}

daily = {
"ts_code":"股票代码",
"trade_date":"交易日期",
"open":"开盘价",
"high":"最高价",
"low":"最低价",
"close":"收盘价",
"pre_close":"昨收价",
"change":"涨跌额",
"pct_chg":"涨跌幅(未复权)",
"vol":"成交量(手)",
"amount":"成交额(千元)",
}



def stock_record():
    # 获取信息
    fields=list(d.keys())
    headers= list (d.values())
    data = pro.query('stock_basic', exchange='', list_status='L', fields=fields)
    #print(data.values)
    data.columns=headers
    data.to_csv("data/stock.csv",header=True,encoding='utf-8',index=False)
    

    # with open("data/stock.csv","w",newline="") as f:
    #     file_writer= csv.writer(f)
    #     file_writer.writerow(headers)
    #     file_writer.writerows(data.values)

def stock_data():
    #初始化跟新时间
    data= pd.read_csv("data/stock.csv",dtype=object)
    # a 股票的全部信息 ，b 要去掉的几列
    a= list(d.values())
    b= ["TS代码","股票名称","上市日期","退市日期"]
    today =datetime.date.today().strftime("%Y%m%d")

    header = [val for val in a if val not in b]
    print(header)
    data_1= data.drop(header,axis=1)
    data_1.insert(loc=len(data_1.columns),column="更新时间",value=today,allow_duplicates=False)
    
    for row in data_1.values:
        row[4]=row[2]
     
    data_1.to_csv("data/data.csv",header=True,encoding='utf-8',index=False)


def dateRange(beginDate,endDate):
    dates=[]
    dt=datetime.datetime.strptime(beginDate,"%Y%m%d")
    date=beginDate[:]
    while date <= endDate:
        dates.append(date)
        dt=dt + datetime.timedelta(1)
        date=dt.strftime("%Y%m%d")
    return dates

def dailyInfoUpdate():
    """
    更新stock 每日信息
    :return:
    """
    df =stock_sql.getTradeDateList()
    #按照symbol 升级
    # df.sort_values(by=["symbol"], inplace=True, ignore_index=True)
    stock_list = np.array(df)

    #多线程支持
    # stock_split = stock_sql.arr_split( stock_list ,500)
    # executor = ProcessPoolExecutor(max_workers=8)
    # all_task = [executor.submit(updateDailyList, (list_a)) for list_a in stock_split]
    # for future in as_completed(all_task) :
    #     res = future.result()

    updateDailyList(stock_list)
    #print(stock_list)

def updateDailyList(stockList):
    """
    :param stockList:输入的股票列表包含 symbol name tscode  lastdate
    :return:
    """
    today = datetime.date.today().strftime("%Y%m%d")
    for item in stockList:
       ## time.sleep(.3)
        print(item)
        data = getStockInfo(item[0],item[2],item[3],"20200312")
        if  data is not None:
            last =stockDateConcat(item[0],data)
            stock_sql.updateTradeDate(item[0],last)

def getStockInfo(symbol, ts_code, start_time,end_time):
    df_list=[]
    while start_time!= end_time:
        df_t = pro.daily(ts_code=ts_code, start_date=start_time, end_date=end_time)
        df_list.append(df_t)
        end_time = df_t.tail(1)["trade_date"].values[0]

    if len(df_list) != 0:
        data = pd.concat(df_list, join="inner")
    else:
        data =None

    return  data

def stockDateConcat(symbol,data):
    """
    将stock 每日新增信息与原始信息合并保存
    :param symbol:
    :param data: 新增数据
    :return:
    """
    path = stock_sql.getFilePath(symbol)
    df=stock_sql.getStockData(path)
    data.trade_date = data.trade_date.astype("int64")
    #新数据放在前
    data_n =pd.concat([data,df],join='inner', ignore_index=True)
    # 1.数据去重
    data_n.drop_duplicates(subset=["trade_date"], keep='first', inplace=True)
    data_n.to_csv(path,header=True,encoding='utf-8',index=False)
    return  data_n.head(1)["trade_date"].values[0]


if __name__ == "__main__":
    #stock_record()
    #stock_data()
    dailyInfoUpdate()
