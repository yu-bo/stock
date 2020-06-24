import  prepare
import stock_sql
import numpy as np
from typing import List

RREDICT_LEN=30

def getHeader():
    """
    获取训练数据的中文头
    :return:
    """
    column_x,_ = prepare.columnSplit()
    headers=[prepare. daily[x]  for x in column_x]
    return headers


def getDateTime(stockInfo,length=RREDICT_LEN):
    if not stockInfo :
        return  np.array()
    symbol = stockInfo["symbol"]

    df = stock_sql.getDailyFrame(symbol)
    x = np.array( df[["trade_date"]])
    return x[-length:]



if __name__ == '__main__':
  pass