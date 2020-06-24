# -*-coding:utf-8 -*-
import sys
#print(sys.executable)
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import  stock_sql
from abc import ABCMeta, abstractmethod
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor,as_completed
import time
import get_info
import  datetime


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


class stockInfo:
    def __init__(self,symbol:str ="000001",name:str="平安银行"):
        self.symbol=symbol
        self.name=name


SEQUENCE_LEN=20
RREDICT_LEN=30

def parseUntrainedData(symbol,df):
    """
    解析未训练用的数据
    :return:
    """
    index =0
    dateInfo =stock_sql.getLastDate(symbol)
    lastDate = int(dateInfo[1])
    for i,row in df.iterrows():
        if row["trade_date"] >= lastDate:
            index  = i
            #df= df.iloc[:index+ SEQUENCE_LEN]
            break
    if index < SEQUENCE_LEN -1 : #数据从未训练过 应该从0开始训练
        index =0
    elif SEQUENCE_LEN -1 <= index: # 数据训练过：
        index = index+1  -(SEQUENCE_LEN -1)
    return index

def getTrianColnum(stockinfo:stockInfo,length=RREDICT_LEN):
    symbol = stockinfo.symbol
    df = stock_sql.getDailyFrame(symbol)
    # 对数据进行训了测试的分离处理
    x, y = prepareLogistics().dataProcess(df)
    col =["close"]
    close = np.array(x[col])[-length:]
    return  np.array( close)

def columnSplit(verify=False):
    """
    将数据按照列的方式拆分成 x和y
    verify :验证y 的日期是否正确 开启时 将带上y对应的日期
    :return:
    """
    all = list(daily.keys());
    column_y=['ts_code']
    column_x = [x for x in all if x not in column_y]
    if verify:# 带上日期 验证 日期是否对应
        column_y = ["close",'trade_date']
    else:
        column_y = ["close"]
    return  column_x ,column_y


def getPredictTrainnData( stockInfo,length =RREDICT_LEN):
    #获取stock 原始信息 ，去重 、预测数据对其
    symbol = stockInfo["symbol"]
    df = stock_sql.getDailyFrame(symbol)
    column_x, column_y = columnSplit()
    x = np.array(df[column_x])
    x= np.around(x,decimals=2)
    return  x[-length:]


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
    return train_x, test_x


class prepareBase:
    def __init__(self):
        print(str( type(self) )+" __build __")
        self.verify = False
        self.only_untrain = False
        self.batch_size = 200
        self.SEQUENCE_LEN = 50
        self.RREDICT_LEN=30
        self.tuple_x=1
        pass

    def getInputShape(self):
        """
        获取训练数据的shape
        :return:  shape
        """
        column_x, _ = columnSplit()
        return (self.SEQUENCE_LEN, len(column_x))


    def dataProcess(self,df:pd.DataFrame)->(np.array,np.array):
        """
        :param df:
        :return:
        """
        # 4.对数据按照表格拆分
        column_x, column_y = columnSplit(self.verify)
        x, y = df[column_x], df[column_y]
        # 5拼凑数据 x的最后一行没有预测值 y的第一行没有 训练值
        x, y = x.drop(len(x) - 1, axis=0), y.drop(0, axis=0)
        y.reset_index(drop=True, inplace=True)
        return x, y

    def dataSequence(self,x:pd.DataFrame,nor:bool=True)->np.array:
        """
        将dataFrame数据 组成训练序列
        :param df: 原始dataFrame数据
        :param nor: 是否对数据标准化
        :param len: 序列长度
        :return: 序列化数据以及 标准化的Scaler
        """
        scaler = MinMaxScaler()
        if nor :
            data_all = np.array(x).astype("float64")
            data_all = scaler.fit_transform(data_all)
        else:
            data_all = np.array(x)
        data = []
        for i in range(len(data_all) - self.SEQUENCE_LEN  + 1):
            data.append(data_all[i: i +  self.SEQUENCE_LEN])
        x = np.array(data).astype('float64')

        return x, scaler

    def dataSequence_y(self,y:pd.DataFrame,nor:bool=True)->np.array:
        """
        将dataFrame数据 组成训练序列
        :param df: 原始dataFrame数据
        :param nor: 是否对数据标准化
        :param len: 序列长度
        :return: 序列化数据以及 标准化的Scaler
        """
        scaler = MinMaxScaler()
        if nor :
            data_all = np.array(y).astype("float64")
            data_all = scaler.fit_transform(data_all)
        else:
            data_all = np.array(y)
        data = []
        for i in range(len(data_all) - self.SEQUENCE_LEN + 1):
            data.append(data_all[i + self.SEQUENCE_LEN - 1])
        x = np.array(data)

        return x, scaler

    def test(self):
        print("+++++++++++++++++++++++")
        self.verify = True
        self.only_untrain = True
        df = stock_sql.getStockFrame()
        df.sort_values(by=["symbol"], inplace=True, ignore_index=True)
        stock_list = np.array(df)
        for info in stock_list:
            data = self.getTrainData(stockInfo(symbol=info[0]))
            data_x, data_y = data[0], data[1]
            for i in range(len(data_x)):
                print(data_x[i][self.SEQUENCE_LEN - 1])
                print(data_y[i])

    def getTrainData(self, stockinfo: stockInfo) -> (np.array, np.array):
        """
           获取单只股票的训练数据
           :param stockInfo: stock 信息
           :param only_untrain: 只使用未训练的数据
           :return: 如果数据不够返回None
           """
        symbol = stockinfo.symbol
        df = stock_sql.getDailyFrame(symbol)
        if len(df) <= self.SEQUENCE_LEN:
            return None
        # 对数据进行训了测试的分离处理
        x, y = self.dataProcess(df)
        train_x, scaler_x = self.dataSequence(x, nor= not self.verify)
        train_y, scaler_y = self.dataSequence_y(y,nor =not self.verify)
        if self.only_untrain:
            index = parseUntrainedData(symbol, df)
            if index >= len(train_x):
                # 起始训练数据超出是长度,没有数据
                return None
            train_x, train_y = train_x[index:], train_y[index:]
        return train_x, train_y,scaler_x,scaler_y

    def getTestData(self, stockInfo:stockInfo):
        data =self.getTrainData(stockInfo)
        if data is not None and len(data[0])>=0:
            train_x, train_y =data[0],data[1]
            return  train_x[-self.RREDICT_LEN:], train_y[-self.RREDICT_LEN:],data[2],data[3]
        else:
            return None

    def dataGenerator(self, index: int = 0):
        df = stock_sql.getStockFrame()
        df.sort_values(by=["symbol"], inplace=True, ignore_index=True)
        # 截取index以后的数据
        stock_array = np.array(df)[index:]
        batch_array = stock_sql.arr_split(stock_array,  self.batch_size)

        for s_array in batch_array:
            # 多进程
            list_x, list_y =  self.dataConcat(s_array)
            # list_x,list_y =dataConcat(s_array,only_untrain)
            if len(list_x[0]) == 0 or len(list_y) == 0: continue
            data_x = []
            for i in range(self.tuple_x):
                data_x.append( np.concatenate(list_x[i], axis=0))
            data_y =  np.concatenate(list_y, axis=0)
            yield data_x, data_y, s_array
            # yield (getTrainData({"symbol":info[0]},only_untrain),info)

    def dataConcat(self,stock_list):
        """
           批量处理stock list 数据  将每只处理的stock数据放到 listx 和 listy中
           :param stock_list:  待处理的stock list 包含symbol name 信息
           :param only_untrain:  只处理未使用的数据
           :return: list_x,list_x
        """
        list_x, list_y = [], []
        for i in range(self.tuple_x):
            list_x.append([])

        for info in stock_list:
            data =self.getTrainData(stockInfo(symbol=info[0]))
            print("stock info :" + info + "\r\n")
            if data == None: continue
            for i in  range(self.tuple_x):
                list_x[i].append(data[0][i])
            list_y.append(data[1])
        return list_x, list_y

    def dataConcatMultiple(self, stock_array, only_untrain=False):
        list_x, list_y = [], []
        for i in range(self.tuple_x):
            list_x.append([])
        split_array = np.array_split(stock_array, 4, axis=0)
        executor = ProcessPoolExecutor(max_workers=4)

        all_task = [executor.submit(self.dataConcat, list_a) for list_a in split_array]
        for future in as_completed(all_task):
            res = future.result()
            for i in range(self.tuple_x):
                list_x[i].extend(res[i][0])
            list_y.extend(res[1])
        return list_x, list_y

class logisticsAllScaler(prepareBase):
    def __init__(self):
        super(logisticsAllScaler,self).__init__()
        self.SEQUENCE_LEN=50
        self.list_x= None
        time.sleep(10)
        pass

    def getDataWithIndex(self, stock_arr, only_untrain=False):
        list_x, list_y, index_list = [], [], []
        trian_map = {}
        index = 0
        for item in stock_arr:
            print(item[1])
            df = stock_sql.getDailyFrame(item[0])
            if only_untrain:
                index = parseUntrainedData(item[0], df)
            # df.drop(columns=["ts_code"],inplace=True)
            x, y = self.dataProcess(df)
            trian_map[item[0]] = index
            list_x.append(x)
            list_y.append(y)
            index_list.append(len(x))
        return list_x, list_y, index_list, trian_map

    def getDataWithIndexMultiple(self,stock_arr, only_untrain=False):
        list_x, list_y, index_list = [], [], []
        trian_map = {}
        a_split = np.array_split(stock_arr, 4, axis=0)
        executor = ProcessPoolExecutor(max_workers=4)
        all_tasks = [executor.submit(self.getDataWithIndex, stockS_arr, only_untrain) for stockS_arr in a_split]
        for furture in as_completed(all_tasks):
            res = furture.result();
            list_xt, list_yt, index_listt, trian_mapt = res[0], res[1], res[2], res[3]
            list_x.extend(list_xt)
            list_y.extend(list_yt)
            index_list.extend(index_listt)
            trian_map.update(trian_mapt)
        return list_x, list_y, index_list, trian_map

    def allScaler(self):
        if self.list_x is not None:
            return self.list_x, self.list_y, self.scaler_x, self.scaler_y, self.train_map

        df = stock_sql.getStockFrame()
        df.sort_values(by=["symbol"], inplace=True, ignore_index=True)
        arr = np.array(df)
        data_new_x, data_new_y, list_index, self.train_map = self.getDataWithIndexMultiple(arr, self.only_untrain)
        data_new_x = np.concatenate(data_new_x, axis=0)
        data_new_y = np.concatenate(data_new_y, axis=0)
        self.scaler_x, self.scaler_y = MinMaxScaler(), MinMaxScaler()
        data_new_x, data_new_y = self.scaler_x.fit_transform(data_new_x), self.scaler_y.fit_transform(data_new_y)
        self.list_x, self.list_y = {}, {}
        list_symbols = list(self.train_map.keys())
        for i in range(len(list_index)):
            data_a, data_b = data_new_x[:list_index[i]], data_new_y[:list_index[i]]
            data_new_x, data_new_y = data_new_x[list_index[i]:], data_new_y[list_index[i]:]
            self.list_x[list_symbols[i]] = data_a
            self.list_y[list_symbols[i]] = data_b

        return self.list_x, self.list_y, self.scaler_x, self.scaler_y, self.train_map

    def getTrainData(self, stockinfo: stockInfo) -> (np.array, np.array):
        self.allScaler()
        symbol= stockinfo.symbol
        x, y = self.list_x[symbol], self.list_y[symbol]
        train_x, _ = self.dataSequence(x, nor=False)
        train_y, _ = self.dataSequence_y(y, nor=False)
        if self.only_untrain:
            index = self. trian_map[symbol]
            if index >= len(train_x):  # 起始训练数据超出是长度,没有数据
                return None
            train_x, train_y = train_x[index:], train_y[index:]
        return train_x,train_y,self.scaler_x,self.scaler_y


class prepareLogistics(prepareBase):
    def __init__(self):
        super(prepareLogistics,self).__init__()
        self.SEQUENCE_LEN=50
        time.sleep(10)
        pass


class prepareClassify(prepareBase):

    def __init__(self):
        super(prepareClassify,self).__init__()
        time.sleep(10)
        pass

    def dataProcess(self,df:pd.DataFrame)-> (np.array,np.array):
        """
        :param df: 包含stock信息的 dataframe
        :return:
        """
        # 4.对数据按照表格拆分
        column_x, column_y = columnSplit(self.verify)
        x, y_t = np.array( df[column_x]),np.array( df[column_y])
        # 5拼凑数据 x的最后一行没有预测值 y的第一行没有 训练值
        y_value= [ y_t[i+1][0]-y_t[i][0] for i in range(len(y_t)-1)]
        y_value= np.int32( np.array( y_value)> 0).reshape(-1, 1)
        if self.verify:
            y_date= [str(y_t[i][1]) + "->" + str(y_t[i + 1][1]) + " :" + str(y_t[i + 1][0]) + "-" + str(y_t[i][0]) for i in range(len(y_t)-1)]
            y_date  = np.array(y_date).reshape(-1, 1)
            y_value = np.concatenate(( y_date ,y_value),axis=1)
        return x[: len(x)-1], y_value

    def dataSequence_y(self,y:pd.DataFrame,nor :bool =False )-> np.array:
        data_all = np.array(y)
        data = []
        for i in range(len(data_all) -self.SEQUENCE_LEN  + 1):
            data.append(data_all[i + self.SEQUENCE_LEN - 1])
        return np.array(data) ,None


class prepareLogistics_Ex(prepareBase):
    def __init__(self):
        super(prepareLogistics_Ex,self).__init__()
        self.SEQUENCE_LEN=50
        self.tuple_x=3
        stock_base = get_info.stock_basic()
        self.area_vec,self.indu_vec ,self.area_shape,self.indu_shape= stock_base.getExVec()
        time.sleep(10)
        pass

    def getInputShape(self):
        daily_shape = super().getInputShape()
        area_shape = (self.area_shape[1],)
        indu_shape =  (self.indu_shape[1],)
        return daily_shape,area_shape,indu_shape

    def getTrainData(self, stockinfo: stockInfo) -> (np.array, np.array):
        data= super().getTrainData(stockinfo)
        if data  is None:
            return  None
        train_x, train_y, self.scaler_x, self.scaler_y = data
        sample_len = train_x.shape[0]
        ex_area ,ex_indu=self.area_vec[stockinfo.symbol],self.indu_vec[stockinfo.symbol]
        ex_area,ex_indu = np.tile(ex_area,(sample_len,1)), np.tile(ex_indu,(sample_len,1))

        train_x =train_x,ex_area,ex_indu

        return train_x,train_y,self.scaler_x,self.scaler_y


if __name__ == "__main__":
    # getTrainData(1)
    # allScaler()
    # for data in dataGenerator(0,True):
    #     print("data")
    pre =prepareClassify()
    pre.test()

