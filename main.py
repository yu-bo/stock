
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras


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



def load_data(df, sequence_length=9, split=0.8):
    #df = pd.read_csv(file_name, sep=',', usecols=[1])
    #data_all = np.array(df).astype(float)
    
    data_all = np.array(df).astype(float)
    scaler = MinMaxScaler()
    data_all = scaler.fit_transform(data_all)
    data = []
    for i in range(len(data_all) - sequence_length - 1):
        data.append(data_all[i: i + sequence_length + 1])
    reshaped_data = np.array(data).astype('float64')
    #np.random.shuffle(reshaped_data)
    # 对x进行统一归一化，而y则不归一化
    # x = reshaped_data[:, :-1]
    # y = reshaped_data[:, ]
    x = reshaped_data
    split_boundary = int(reshaped_data.shape[0] * split)
    train_x = x[: split_boundary]
    test_x = x[split_boundary:]

    #train_y = y[: split_boundary]
    #test_y = y[split_boundary:]

    return train_x, test_x, scaler



def load_info():
    name ="阿科力-603722.SH"
    fileName = "".join(["data/info/",name,".csv"])
    data = pd.read_csv(fileName)
    cols = list( daily.keys())[2:]
    dd=data[cols]
    #print(data)
    return dd

def data_prepare():
    data_x = load_info()
    train_x, test_x,_ = load_data(data_x)
    data_y =data_x[["close"]]
    train_y, test_y,_ = load_data(data_y)
    return train_x,train_y,test_x,test_y

def build_model():
    model =keras.Sequential()
    model.add(keras.Input(shape=(10,9)))
    model.add(keras.layers.LSTM(units=9,return_sequences=True))
    model.add(keras.layers.LSTM(1, activation='sigmoid', return_sequences=False))
    model.compile(optimizer=keras.optimizers.Adam(),
             loss=keras.losses.BinaryCrossentropy(),
             metrics=['accuracy'])
    model.summary()
    return model

def trainning():
    train_x,train_y,test_x,test_y = data_prepare()
    print(train_x.shape, train_y.shape)
    model= build_model()

    history = model.fit(train_x, train_y,
                        validation_data=(test_x, test_y),
                        batch_size=64,
                        epochs=5)



if __name__ == "__main__":
    #load_info()
    trainning()
    #get_model()
   

