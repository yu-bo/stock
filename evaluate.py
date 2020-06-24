import  math
import  numpy as np
from scipy import stats
from  sklearn import metrics
import trainning
import stock_sql
import prepare
import matplotlib.pyplot as plt

from   prepare import  stockInfo

testX = [174.5, 171.2, 172.9, 161.6, 123.6, 112.1, 107.1, 98.6, 98.7, 97.5, 95.8, 93.5, 91.1, 85.2, 75.6, 72.7, 68.6,
         69.1, 63.8, 60.1, 65.2, 71, 75.8, 77.8]
testY = [88.3, 87.1, 88.7, 85.8, 89.4, 88, 83.7, 73.2, 71.6, 71, 71.2, 70.5, 69.2, 65.1, 54.8, 56.7, 62, 68.2, 71.1,
         76.1, 79.8, 80.9, 83.7, 85.8]


def explained_variance_score(y_true,y_pred):
    """
    解释方差分，这个指标用来衡量我们模型对数据集波动的解释程度，如果取值为1时，模型就完美，越小效果就越差.
    :param y_true:实际值
    :param y_pred:预测值
    :return:
    """
    socre =  metrics.explained_variance_score(y_true,y_pred)
    socre = round(socre,4)
    return socre

def mean_absolute_error(y_true,y_pred):
    """
    平均绝对误差,给定数据点的平均绝对误差，一般来说取值越小，模型的拟合效果就越好
    :param y_true:
    :param y_pred:
    :return:
    """
    socre = metrics.mean_absolute_error(y_true, y_pred)
    socre = round(socre, 4)
    return socre

def mean_squared_error(y_true,y_pred):
    """
    均方误差
    :param y_true:
    :param y_pred:
    :return:
    """
    socre = metrics.mean_squared_error(y_true, y_pred)
    socre = round(socre, 4)
    return socre

def mean_squared_log_error(y_true,y_pred):
    socre = metrics.mean_squared_log_error(y_true, y_pred)
    socre = round(socre, 4)
    return socre


def r2_socre(y_true,y_pred,  len=4):
    """
    R方可以理解为因变量y中的变异性能能够被估计的多元回归方程解释的比例，
    它衡量各个自变量对因变量变动的解释程度，
    其取值在0与1之间，其值越接近1，则变量的解释程度就越高，其值越接近0，其解释程度就越弱。
    :param y_true:
    :param y_pred:
    :return:
    """
    socre = metrics.r2_score(y_true, y_pred)
    socre = round(socre, len)
    return socre

def computeCorrelation(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2

    SST = math.sqrt(varX * varY)
    print("使用math库：r：", SSR / SST, "r-squared：", (SSR / SST) ** 2)
    socre = round(SST, 4)
    return socre

def upDownfit(today, real_t,pred_t):
    real_up ,pred_up= real_t -today ,pred_t -today
    real_up,pred_up=np.round(real_up,decimals=2),np.round(pred_up,decimals=2)
    same = np.multiply(real_up,pred_up)
    count =0
    for i in range(len(same)):
        if same[i] > 0:
            count += 1
    #print( count)
    index =np.maximum(same,0)
    return  real_up,pred_up,count,len(same)

def evaluation():
    realData, predictData = trainning.testing({"symbol":"000001"})
    score = r2_socre(realData,predictData)
    print(score)


class classifyEvaluate:
    def __init__(self):
        self.trainMode = trainning. classifyTrain_1()
        pass

    def fitCount(self,realD:np.array,preD:np.array)-> int:
        count=0
        for i in range(len( realD)):
            if realD[i]==preD[i]:
                count+=1
        return count

    def ClearRecord(self):
        stock_sql.ClearRecordData()
    def upDownrate(self):

        df = stock_sql.getStockFrame()
        stockList = np.array(df)
        fitConunt = {}
        for i in range(prepare.RREDICT_LEN + 1):
            fitConunt[str(i)] = 0
        for stock in stockList[0:]:
            realData, predictData = self.trainMode.predict(stockInfo(symbol=stock[0]))
            predictData= [np.argmax(predictData[i]) for i in range(len(predictData))]
            predictData =np.array(predictData).reshape((-1,1))
            if realData.shape[0] != prepare.RREDICT_LEN:
                continue
            count= self.fitCount(realData,predictData)
            print("symbol:" + str(stock[0]) + "  name:" + str(stock[1]) + "  fit:" + str(count))
            fitConunt[str(count)] += 1
        x = [i for i in range(prepare.RREDICT_LEN + 1)]
        y = [fitConunt[str(d)] for d in range(prepare.RREDICT_LEN + 1)]
        plt.plot(x, y)
        plt.show()
        print(fitConunt)

class logisticsEvalutate:
    def __init__(self):
        self.trainMode= trainning. logisticsTrain()
        pass
    def upDownrate(self):
        stock_sql.ClearRecordData()
        df = stock_sql.getStockFrame()
        stockList = np.array(df)
        fitConunt = {}
        for i in range(prepare.RREDICT_LEN + 1):
            fitConunt[str(i)] = 0
        for stock in stockList[0:]:
            realData, predictData = self.trainMode.predict(stockInfo(symbol=stock[0]))
            if realData.shape[0] != prepare.RREDICT_LEN:
                continue
            closeColnum = prepare.getTrianColnum(stockInfo(symbol=stock[0]))
            real_up, pred_up, count, total = upDownfit(closeColnum, realData, predictData)
            print("symbol:" + str(stock[0]) + "  name:" + str(stock[1]) + "  fit:" + str(count))
            fitConunt[str(count)] += 1
        x = [i for i in range(prepare.RREDICT_LEN + 1)]
        y = [fitConunt[str(d)] for d in range(prepare.RREDICT_LEN + 1)]
        plt.plot(x, y)
        plt.show()
        print(fitConunt)

    def RRrate(self):
        stock_sql.ClearRecordData()
        df = stock_sql.getStockFrame()
        stockList = np.array(df)
        fitConunt = {}

        for stock in stockList[0:]:
            realData, predictData = trainning. logisticsTrain().predict(stockInfo(symbol=stock[0]))
            if len(realData) is 0:
                continue
            r2 = r2_socre(realData, predictData, 0)
            if r2 in fitConunt.keys():
                fitConunt[r2] += 1
            else:
                fitConunt[r2] = 0
            print("symbol:" + str(stock[0]) + "  name:" + str(stock[1]) + "  fit:" + str(r2))

        x = list(fitConunt.keys())
        x.sort()
        y = [fitConunt[d] for d in x]
        x = [math.tanh(d) for d in x]
        plt.plot(x, y)
        plt.show()
        print(fitConunt)

def xxxx():
    aa =np.array([i for i in range(130)])
    x = prepare.dataSequence_y(aa,nor=False)
    print(aa)

if __name__ == '__main__':
    eva= classifyEvaluate()
    eva.ClearRecord()
    eva.upDownrate()

    pass
    #xxxx()