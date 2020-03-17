# -*-coding:utf-8 -*-
import sys
#print(sys.executable)
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import  stock_sql

import  socket

# HOST='192.168.8.120'
# PORT=8021
# s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# s.connect((HOST,PORT))
# data=s.recv(1024)
# print ("Received:",data)
# s.close()


#pd.show_versions()
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



weight_file = "".join(["model/weights/my_weight"])
index_file = "index.txt"

SEQUENCE_LEN=20
INPUT_LEN=10
RREDICT_LEN=30

MY_FILE_NAME="data/info/天地科技-600582.SH.csv"

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
# print(gpus)
# print(cpus)
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
#tf.config.experimental.set_visible_devices(devices=cpus[0], device_type='CPU')
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[1],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1536)]
# )

def trainDataGenerator(stockInfo,batch_size=512):
    """
    匹配keras fit generator
    :param stockInfo:
    :param batch_size:
    :return:
    """
    train_x,scaler_x,train_y, scaler_y = getTrainData(stockInfo)
    block = train_x.shape[0] // batch_size;
    for i in range(block + 1):
        x, y = train_x[i * batch_size:i * batch_size + batch_size], train_y[i * batch_size:i * batch_size + batch_size]
        print(x.shape)
        yield (x, y)

def parseUntrainedData(symbol,df):
    """
    解析未训练用的数据
    :return:
    """
    dateInfo =stock_sql.getLastDate(symbol)
    lastDate = int(dateInfo[1])
    for index,row in df.iterrows():
        if row["trade_date"] <= lastDate:
            df= df.iloc[:index+ SEQUENCE_LEN]
            break
    return df


def getTrainData(stockInfo,only_untrain=False):
    """
    获取单只股票的训练数据
    :param stockInfo: stock 信息
    :param only_untrain: 只使用未训练的数据
    :return: 如果数据不够返回None
    """
    symbol = stockInfo["symbol"]
    path = stock_sql.getFilePath(symbol)
    df=stock_sql.getStockData(path)

    if only_untrain :
        df= parseUntrainedData(symbol,df)
    if len(df) <= SEQUENCE_LEN:
        return None

    # 对数据进行训了测试的分离处理
    x,y = dataProcess(df)
    #形成训练数据
    train_x, scaler_x = dataSequence(x)
    # y只取最后一位
    train_y, scaler_y = dataSequence_y(y)

    return train_x,scaler_x,train_y, scaler_y

def getTestData(stockInfo,length=RREDICT_LEN):
    train_x, scaler_x, train_y, scaler_y =getTrainData(stockInfo)
    return  train_x[-length:], scaler_x, train_y[-length:], scaler_y


def datetimeProcess(df):
    df =df[["trade_date"]]
    df.sort_values(by=["trade_date"],ascending=True,ignore_index=True, inplace=True)
    x= np.array(df)
    return x

def predictDataProcess(df):
    df.fillna(0,inplace=True)
    df.sort_values(by=["trade_date"],ascending=True,ignore_index=True,inplace=True)
    column_x, column_y = columnSplit()
    x =np.array(df[column_x])
    return  x

def dataProcess(df):

    #2.将nan数据 替换为0
    df.fillna(0,inplace=True)
    #3.按照日期排序 ,并忽略索引
    df.sort_values(by=["trade_date"],ascending=True,ignore_index=True, inplace=True)
    #4.对数据按照表格拆分
    column_x, column_y = columnSplit()
    x,y = df[column_x],df[column_y]
    #5拼凑数据 x的最后一行没有预测值 y的第一行没有 训练值
    x, y = x.drop(len(x)-1,axis=0), y.drop(0,axis=0)
    y.reset_index(drop=True, inplace=True)
    # x.insert(loc=0,column='close',value=list(y["close"]))
    # x.insert(loc=0,column="aa",value=list(y["trade_date"]))
    return  x,y


def dataSequence(df,nor=True,seq_len=SEQUENCE_LEN):
    """
    将dataFrame数据 组成训练序列
    :param df: 原始dataFrame数据
    :param nor: 是否对数据标准化
    :param len: 序列长度
    :return: 序列化数据以及 标准化的Scaler
    """
    scaler = MinMaxScaler()

    data_all = np.array(df).astype(float)
    if nor:
        data_all = scaler.fit_transform(data_all)
    data = []
    for i in range(len(data_all) - seq_len):
        data.append(data_all[i: i + seq_len])
    x = np.array(data).astype('float64')

    return x, scaler

def dataSequence_y(df,nor=True,seq_len=SEQUENCE_LEN):
    """
    将dataFrame数据 组成训练序列
    :param df: 原始dataFrame数据
    :param nor: 是否对数据标准化
    :param len: 序列长度
    :return: 序列化数据以及 标准化的Scaler
    """
    scaler = MinMaxScaler()

    data_all = np.array(df).astype(float)
    if nor:
        data_all = scaler.fit_transform(data_all)
    data = []
    for i in range(len(data_all) - seq_len):
        data.append(data_all[i + seq_len])
    x = np.array(data).astype('float64')

    return x, scaler


def columnSplit():
    """
    将数据按照列的方式拆分成 x和y
    :return:
    """
    all = list(daily.keys());
    column_y=['close',"ts_code"]
    column_x = [x for x in all if x not in column_y]
    column_y = ['close']

    return  column_x ,column_y


def getDateTime(stockInfo,length=RREDICT_LEN):
    if not stockInfo :
        return  np.array()
    symbol = stockInfo["symbol"]

    path = stock_sql.getFilePath(symbol)
    df = stock_sql.getStockData(path)
    x= datetimeProcess(df)
    return x[-length:]

def getPredictTrainnData( stockInfo,length =RREDICT_LEN):
    #获取stock 原始信息 ，去重 、预测数据对其
    symbol = stockInfo["symbol"]
    path = stock_sql.getFilePath(symbol)
    df = stock_sql.getStockData(path)
    x= predictDataProcess(df)
    x= np.around(x,decimals=2)
    return  x[-length:]

def getHeader():
    """
    获取训练数据的中文头
    :return:
    """
    column_x,_ =columnSplit()
    headers=[daily[x]  for x in column_x]
    return headers

def getInputShape():
    """
    获取模型输入shape
    :return:
    """
    column_x,_= columnSplit()
    return (SEQUENCE_LEN,len(column_x))

def updateTrianRecord(stock_list):#480
    """
    跟新股票的 训练日期
    :param stock_list: symbol name
    :return:
    """

    print(stock_list)
    path_list = stock_sql.getPathList(stock_list) #(path,symbol)
    up_list =stock_sql.getLastTradeList(path_list)# symbol date tscode
    stock_sql. updateTrianList(up_list)
    print(up_list)

def dataConcat(stock_list,only_untrain =False):
    """
    批量处理stock list 数据
    :param stock_list:  待处理的stock list 包含symbol name 信息
    :param only_untrain:  只处理未使用的数据
    :return: data_X,data_Y
    """
    list_x ,list_y=[],[]
    for info in stock_list:
        data = getTrainData({"symbol": info[0]}, only_untrain)
        print("stock info :" + info + "\r\n")
        if data == None: continue
        list_x.append(data[0])
        list_y.append(data[2])

    if len(list_x) ==  0 or len(list_y)==0:return  None
    data_x,data_y = np.concatenate(list_x,axis=0),np.concatenate(list_y,axis=0)
    return  data_x,data_y

def dataGenerator(index = 0,only_untrain=False,batch_size =200):
    """
    数据生成器，返回x，y ，stocklist
    :param index:
    :param only_untrain:
    :param batch_size:
    :return:  data_X,data_y, stock_list
    """
    df =stock_sql.getStockList()
    df.sort_values(by=["symbol"],inplace=True,ignore_index=True)
    #截取index以后的数据
    stock_list =np.array(df)[index:]
    batch_list =stock_sql.arr_split(stock_list,batch_size)

    for s_list in batch_list:
        data =dataConcat(s_list,only_untrain)
        if data is None : continue
        yield  data[0],data[1],s_list

       # yield (getTrainData({"symbol":info[0]},only_untrain),info)



def dataSplit(npArr, split=0.9):
    """
    :param npArr:
    :param split:
    :return:
    """
    #对数据进行拆分 获得验证数据集
    split_boundary = int(npArr.shape[0] * split)
    train_x = npArr[: split_boundary]
    test_x = npArr[split_boundary:]

    # train_y = y[: split_boundary]
    # test_y = y[split_boundary:]
    return train_x, test_x



if __name__ == "__main__":

    for data in dataGenerator(0):
        print("data")

