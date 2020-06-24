from datetime import datetime
import pandas as pd
import numpy as np
import sqlite3
import os
import pickle

d = {"ts_code": "TS代码",
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

DataBase = "data\my_stock.db"
RECORDTABLE = ''


class my_sql:

    def __init__(self, baseName):
        self.conn = sqlite3.connect(baseName)

    def __enter__(self):
        # print('__enter__() is call!')
        return self.conn.cursor()

    def __exit__(self, exc_type, exc_value, traceback):
        self.conn.commit()
        self.conn.close()
        # print('__exit__() is call!')
        # print(f'type:{exc_type} : ' + f'value:{exc_value}')
        # print(f'value:{exc_value}')
        # print(f'trace:{traceback}')
        return True  # 异常不抛出


def CreateStockTabel():
    Sql_str = 'CREATE TABLE stock_info ('
    for item in list(d.keys()):
        Sql_str += item + " CHAR(50), "
    Sql_str = Sql_str[:-2]
    Sql_str += ") ;"

    print(Sql_str)
    with my_sql(DataBase) as c:
        c.execute(Sql_str)


def createRecordTable():
    Sql_str = 'CREATE TABLE  IF NOT EXISTS stock_record (' \
              'symbol TEXT PRIMARY KEY, name TEXT, time TEXT, real_data BLOB, predict_data BLOB )'
    with my_sql(DataBase) as c:
        c.execute(Sql_str)

def createDateTable():
    """
    创建stock_date 数据表 用来记录daily  train等日期信息
    :return:
    """
    Sql_str = 'CREATE TABLE  IF NOT EXISTS stock_date (' \
              'symbol TEXT PRIMARY KEY, name TEXT, ts_code TEXT, ' \
              'list_date TEXT, trade_date TEXT, train_date TEXT)'

    with my_sql(DataBase) as c:
        c.execute(Sql_str)

def saveStockData():
    """
    将stock的基本信息存放再数据库
    :return:
    """
    df = pd.read_csv("data/stock_basic.csv", dtype=object)
    df.columns = list(d.keys())
    conn = sqlite3.connect(DataBase)
    df.to_sql(name="stock_info", con=conn, index=False, if_exists="replace")
    conn.close()


def initDateTable():
    """
    生成 表用来记录 stock信息更新到哪一天， 模型对训练到哪一天
    :return:
    """
    sqlStr = "SELECT symbol, name, ts_code,list_date FROM stock_info "
    conn = sqlite3.connect(DataBase)
    df = pd.read_sql(sqlStr, con=conn)
    conn.close()
    with my_sql(DataBase) as c:
        for index, row in df.iterrows():
            sqlStr = "REPLACE INTO stock_date (symbol, name, ts_code, list_date, info_date, train_date) " \
                     "VALUES (?,?,?,?,?,?)"
            c.execute(sqlStr,(row["symbol"],row["name"],row["ts_code"],
                              row["list_date"],row["list_date"],row["list_date"]))



def saveStockPath():
    """
    将stock的位置信息存入数据库
    :return:
    """
    sqlStr = "SElECT name, symbol, ts_code FROM stock_info "
    conn = sqlite3.connect(DataBase)
    df = pd.read_sql(sqlStr, con=conn)

    paths = []
    for item in df.values:
        path_o = "".join(["data/info/", item[0].replace("*", ""), "-", item[2], ".csv"])
        path_w = "".join(["data/info/", item[1], ".csv"])
        paths.append(path_w)
        # os.rename(path_o,path_w)
    df.drop(columns=["ts_code"], inplace=True)
    df.insert(loc=2, column="path", value=paths)
    df.to_sql(name="stock_path", con=conn, index=False, if_exists='replace')
    conn.close()
    print(df)

def initTrainDate():
    """
    初始化 训练日期
    :param symbol:
    :param date:
    :return:
    """
    sqlStr = "UPDATE stock_date SET train_date = list_date"
    with my_sql(DataBase) as c:
        c.execute(sqlStr)
    print("initTrainDate")

def updateTradeDate(symbol,date):
    """
    更新stock的 日期信息 记录日线更新到哪那一天
    :param symbol: stock 代码
    :param date:   交易日期
    :return:
    """
    sqlStr = "UPDATE stock_date SET trade_date ='{}' WHERE symbol='{}'".format(date, symbol)
    with my_sql(DataBase) as c:
        c.execute(sqlStr)


def updateTrianList(stockList):
    """
    更新股票的 训练日期
    :param dateList:输入待跟新的stock 列表  包含 symbol, name
    :return:
    """
    with my_sql(DataBase) as c:
        for item in stockList:  # symbol date tscode
            sqlStr = "UPDATE stock_date SET train_date = trade_date WHERE symbol='{}'".format(item[0])
            c.execute(sqlStr)

def getPathList(symbolList):
    """
    获取stock 信息的文件位置列表
    :param symbolList: 输入列表包含可 symbol和 name 信息
    :return: pathList  输出列表 包含 path 和 symbol
    """
    pathList=[]
    with my_sql(DataBase) as c:
        for item in symbolList: #symbol :(symbol,name)
            sqlStr = "SELECT path FROM stock_path WHERE symbol ='{}'".format(item[0])
            cursor = c.execute(sqlStr)
            path = cursor.fetchone()[0]
            pathList.append((path,item[0]))
    return pathList #(path,symbol)

def getFilePath(symbol):
    sqlStr = "SELECT path FROM stock_path WHERE symbol = \"" + symbol + "\""
    with my_sql(DataBase) as c:
        cursor = c.execute(sqlStr)
        path = cursor.fetchone()[0]
        return path

def getLastTradeList(pathList):
    """
    获取stock 最后更新信息
    :param pathList:输出列表 包含 path 和 symbol
    :return:dateList 输出列表 包含 symbol date tscode)
    """
    dateList =[] # (symbol date tscode)
    for item in pathList: #item : (path, symbol)
        df = pd.read_csv(item[0], dtype=object)
        df.sort_values(by=["trade_date"], inplace=True, ignore_index=True)
        date = df.tail(1)[["trade_date", "ts_code"]].values[0]
        dateList.append((item[1],date[0],date[1]))
    return  dateList

def getLastDate(symbol):
    """
    获取stock的训练数据的最后日期：
    :param symbol: stock的symbol信息
    :return:  : trade_date,train_date
    """
    columns = ["symbol", "trade_date", "train_date"]
    sqlStr = "SELECT {} From stock_date WHERE symbol= '{}'".format(columnsToSql(columns),symbol)

    with my_sql(DataBase) as c:
        cursor= c.execute(sqlStr)
        data = cursor.fetchone()
        return  data[1],data[2]

def getLastTradeDate(symbol):
    """
    获取stock的训练数据的最后日期：
    :param symbol: stock的symbol信息
    :return:  nparray: [date,tscode]
    """
    path =getFilePath(symbol)
    df =pd.read_csv(path,dtype=object)
    df.sort_values(by=["trade_date"],inplace=True,ignore_index=True)
    date = df.tail(1)[["trade_date","ts_code"]]
    return date.values[0]

def getTradeDateList():
    sqlStr= "SELECT symbol, name, ts_code,trade_date FROM stock_date"
    conn =sqlite3.connect(DataBase)
    df= pd.read_sql(sqlStr,con=conn)
    conn.close();
    return df

def getStockFrame(headers = ["symbol","name"]):
    sqlStr = "SELECT {} FROM stock_info".format(columnsToSql(headers))
    conn = sqlite3.connect(DataBase)
    df = pd.read_sql(sqlStr, con=conn, columns=headers)
    conn.close();
    return df

def getStockInfo(item):
    if not item:
        return pd.DataFrame()

    columns = ["symbol", "area", "industry", "market", "list_date", "is_hs"]
    sqlStr = "SELECT {} From stock_info WHERE symbol= '{}'".format(columnsToSql(columns),item["symbol"])
    conn = sqlite3.connect(DataBase)
    df = pd.read_sql(sqlStr, con=conn)
    conn.close()
    df.fillna("", inplace=True)

    if (df.values.shape[0] != 1):
        return pd.DataFrame()
    for index, key, in enumerate(columns):
        name, value = parseStockInfo(key, df.values[0][index])
        print(name, value)
        df.values[0][index] = name + ": " + value
    return df

def getDailyFrame(symbol:str)->pd.DataFrame:
    path = getFilePath(symbol)
    df = getStockData(path)
    # 2.将nan数据 替换为0
    df.fillna(0, inplace=True)
    # 3.按照日期排序 ,并忽略索引
    df.sort_values(by=["trade_date"], ascending=True, ignore_index=True, inplace=True)
    return df

def getStockData(path):
    data = pd.read_csv(path, index_col=False, na_values=0)
    # 1.数据去重
    data.drop_duplicates(subset=["trade_date"], keep='first', inplace=True)
    return data


def getRecordData(symbol):
    if not symbol :
        return None
    now_date = datetime.now().strftime("%Y%m%d")
    sqlStr = "SELECT real_data, predict_data, symbol FROM stock_record" \
             " WHERE symbol='{}' AND time='{}'".format(symbol,now_date)
    with my_sql(DataBase) as c:
        cursor = c.execute(sqlStr)
        data = cursor.fetchone()
        if data :
            real, predict = pickle.loads(data[0]),pickle.loads(data[1])
            return (real,predict)


def SaveRecordData(stockInfo, realData, predictData):
    reald, pred = pickle.dumps(realData), pickle.dumps(predictData)
    now_date = datetime.now().strftime("%Y%m%d")

    sqlStr = "REPLACE INTO stock_record (symbol,name,time,real_data, predict_data) VALUES (?,?,?,?,?) "
    with my_sql(DataBase) as c:
        c.execute(sqlStr, (stockInfo["symbol"], "", now_date, reald, pred))
    #print(now_date)
    pass

def ClearRecordData():
    sqlStr ="DROP TABLE stock_record"
    with my_sql(DataBase) as c:
        c.execute(sqlStr)


def parseStockInfo(key, value):
    name = d[key]
    if key == 'market':
        name = "市场类型"
    elif key == "curr_type":
        if value == "CNY":
            value = "RMB"
    elif key == "list_status":
        name = "上市状态"
        if value == "L":
            value = "上市"
        elif value == "D":
            value = "退市"
        elif value == "P":
            value = "暂停上市"
    elif key == "is_hs":
        name = "沪深港通"
        if value == "S":
            value = "深港通"
        elif value == "H":
            value = "沪港通"
        elif value == "N":
            value = "否"
    return name, value

def columnsToSql(columns):
    """
    把列表转换成对应的sql 列字符串
    :param columns:
    :return:
    """
    columnStr = ""
    for x in columns:
        columnStr += x + ","
    return  columnStr[:-1]


def arr_split(arr,size):
    s = [arr[i:i+size]  for i in range(0,int(len(arr))+1,size)]
    return s



def DataBaseInit():
    # CreateStockTabel()
    saveStockData()


def sqlTest():
    con = sqlite3.connect("data\my_stock.db")
    sql = 'select * from user_information LIMIT 3'
    df = pd.read_sql(sql, con)
    print(df)


if __name__ == '__main__':
    # DataBaseInit()
    # getStockList()
    # saveStockPath()
    initTrainDate()
    #getFilePath("000001")
