import  tushare as ts
import csv
import time
import datetime
import os
import numpy as np
import pandas as pd

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
    a= list(d.values())
    b= ["TS代码","上市日期","退市日期"]
    today =datetime.date.today().strftime("%Y%m%d")

    header = [val for val in a if val not in b]
    print(header)
    data_1= data.drop(header,axis=1)
    data_1.insert(loc=len(data_1.columns),column="跟新时间",value=today,allow_duplicates=False)
    
    for row in data_1.values:
        row[3]=row[1]
     
    data_1.to_csv("data/data.csv",header=True,encoding='utf-8',index=False)


def get_info():
    data = pd.read_csv("data/data.csv", dtype=object)
    today= datetime.date.today().strftime("%Y%m%d")
    for row in data.values:
        start_time =row[len(row)-1]
        end_time =today
        code = row[0]
        df =pro.daily(ts_code=code, start_date=start_time, end_date=end_time)
        print(df)


if __name__ == "__main__":
    #stock_record()
    #stock_data()
    get_info()