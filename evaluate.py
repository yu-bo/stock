import  math
import  numpy as np
from scipy import stats
from  sklearn import metrics
import trainning


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
    return socre

def mean_absolute_error(y_true,y_pred):
    """
    平均绝对误差,给定数据点的平均绝对误差，一般来说取值越小，模型的拟合效果就越好
    :param y_true:
    :param y_pred:
    :return:
    """
    socre = metrics.mean_absolute_error(y_true, y_pred)
    return socre

def mean_squared_error(y_true,y_pred):
    """
    均方误差
    :param y_true:
    :param y_pred:
    :return:
    """
    socre = metrics.mean_squared_error(y_true, y_pred)
    return socre

def mean_squared_log_error(y_true,y_pred):
    socre = metrics.mean_squared_log_error(y_true, y_pred)
    return socre


def r2_socre(y_true,y_pred):
    """
    R方可以理解为因变量y中的变异性能能够被估计的多元回归方程解释的比例，
    它衡量各个自变量对因变量变动的解释程度，
    其取值在0与1之间，其值越接近1，则变量的解释程度就越高，其值越接近0，其解释程度就越弱。
    :param y_true:
    :param y_pred:
    :return:
    """
    socre = metrics.r2_score(y_true, y_pred)
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
    return


def evaluation():
    realData, predictData = trainning.testing({"symbol":"000001"})
    score = r2_socre(realData,predictData)
    print(score)
if __name__ == '__main__':
    evaluation()