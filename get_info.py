import  tushare as ts
import csv
import time
import datetime
import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed


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
    a= list(d.values())
    b= ["TS代码","股票名称","上市日期","退市日期"]
    today =datetime.date.today().strftime("%Y%m%d")

    header = [val for val in a if val not in b]
    print(header)
    data_1= data.drop(header,axis=1)
    data_1.insert(loc=len(data_1.columns),column="跟新时间",value=today,allow_duplicates=False)
    
    for row in data_1.values:
        row[4]=row[2]
     
    data_1.to_csv("data/data.csv",header=True,encoding='utf-8',index=False)


def get_info():
    data = pd.read_csv("data/data.csv", dtype=object)
    today= datetime.date.today().strftime("%Y%m%d")
    headers = data.columns.values

    list_n = []
    list_s = arr_split(data.values, 500)
    executor = ProcessPoolExecutor(max_workers=8)
    
    # all_task = [executor.submit(get_info_list, (list_a)) for list_a in list_s]
    # for future in as_completed(all_task) :
    #     list_tmp = future.result()
    #     list_n +=list_tmp 

    for list_a in list_s:
        list_tmp = get_info_list(list_a)
        list_n += list_tmp
    
    data_n = pd.DataFrame(list_n)
    data_n.columns = headers
    data_n.to_csv("data/data.csv",header=True,encoding='utf-8',index=False)
    #values_n = pd.concat(list_n,join="inner")
    print(data_n)

def get_info_list(arr):
    row_list=[]
    for row in arr:
        time.sleep(0.5)
        row_n= get_daily_info(row)
        row_list.append(row_n)
    return row_list

def get_daily_info(row):
        start_time = row[len(row)-1]
        end_time= datetime.date.today().strftime("%Y%m%d")
        df_list=[]
        lenght=0
        while True:
            df_t = pro.daily(ts_code=row[0], start_date=start_time, end_date=end_time)
            df_list.append(df_t)
            mid_time = df_t.tail(1)["trade_date"].values[0]
            lenght+= df_t.shape[0]
            if mid_time == end_time:
                print(row[0]+"：  end")
                break
            end_time = mid_time
        data= pd.concat(df_list,join="inner")
        row[len(row)-1]= datetime.date.today().strftime("%Y%m%d")
        daily_store(data,row)
        print(data.shape[0])
        return row
       
def get_pre_info(row):
    file_name = "".join(["data/info/",row[1].replace("*",""),"-",row[0],".csv"]) 
    data= pd.read_csv(file_name)
    return data

def arr_split(arr,size):
    s=[]
    for i in range(0,int(len(arr))+1,size):
        c=arr[i:i+size]
        s.append(c)
    return s

def daily_store(data,row):
    file_name = "".join(["data/info/",row[1].replace("*",""),"-",row[0],".csv"])
    data.to_csv(file_name,header=True,encoding='utf-8',index=False)
    print(file_name)
    pass 

if __name__ == "__main__":
    #stock_record()
    #stock_data()
    get_info()
