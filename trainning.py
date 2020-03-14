import sys
#print(sys.executable)
import numpy as np
import matplotlib.pyplot as plt

 
from tensorflow.keras.callbacks import  TensorBoard
from tensorflow.keras.callbacks import  ModelCheckpoint
import datetime
import tensorflow.keras as keras
import prepare
import stock_sql

import pickle




weight_file = "".join(["model/weights/my_weight"])
index_file = "index.txt"
log_dir = "".join(["log\\model_train\\", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")])

def getTrainIndex():
    with open(index_file,"r") as f:
        index=f.readline()
    return  int(index)
def saveTrainIndex(index):
    with open(index_file,"w") as f:
        f.write(str(index))
        f.flush()



def build_model(shape):
    return  model_1(shape)

def model_1(shape):
    model = keras.Sequential()
    model.add(keras.Input(shape=shape))
    #model.add(keras.layers.Dense(units=100, activation="tanh"))
    model.add(keras.layers.LSTM(units=500, activation='tanh',return_sequences=True))
    model.add(keras.layers.LSTM(units=500, activation='tanh', return_sequences=True))
    model.add(keras.layers.LSTM(units=200,activation='tanh',return_sequences= False))
    model.add(keras.layers.Dense(units=200, activation="tanh"))
    model.add(keras.layers.Dense(units=20, activation="tanh"))
    model.add(keras.layers.Dense(units=1))
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss="mse",
                  metrics=[keras.metrics.mae])
    model.summary()

    return model

def model_2(shape):
    model = keras.Sequential()
    model.add(keras.Input(shape=shape))
    model.add(keras.layers.LSTM(units=50, activation='tanh', return_sequences=True))
    model.add(keras.layers.LSTM(units=50, activation='tanh', return_sequences=True))
    model.add(keras.layers.Dense(units=10, activation="tanh"))
    model.add(keras.layers.Dense(units=1))
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss="mse",
                  metrics=[keras.metrics.mae])
    model.summary()
    return model

def trainningOne():
    train_x, scaler_x,train_y,scaler_y = prepare.getTrainData({"symbol":"000001"})
    input = (train_x.shape[1],train_x.shape[2])
    model =build_model(input)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1,  write_graph=True)
    modelcheck_callback =ModelCheckpoint(filepath=weight_file,save_weights_only=True)
    history = model.fit(train_x, train_y, batch_size=1024, epochs=30, validation_split=0.1,
                        callbacks=[tensorboard_callback,modelcheck_callback])


def trainningAll( only_untrain = False):

    model =build_model(shape= prepare.getInputShape())
    try:
        model.load_weights(weight_file)
    except :
        pass
    for data in prepare.dataGenerator(0,only_untrain):# (train_data,stock_info)
        train_x,  train_y, stock_list =data
        history = model.fit(train_x, train_y, batch_size=1024, epochs=30, validation_split=0.05)
        model.save_weights(weight_file)
        stock_sql. updateTrianList(stock_list)

def testing( stockInfo ):

    data =stock_sql.getRecordData(stockInfo)
    if  data:
        return data[0],data[1]
    test_x, scaler_x, test_y, scaler_y = prepare.getTestData(stockInfo)
    input = (test_x.shape[1],test_x.shape[2])
    model =build_model(input)
    model.load_weights(weight_file)
    predict_y =model.predict(test_x)
    realData=scaler_y.inverse_transform(test_y)
    predictData=scaler_y.inverse_transform(predict_y.astype("float64"))
    realData,predictData =np.around(realData,decimals=2),np.around(predictData,decimals=2)

    stock_sql.SaveRecordData(stockInfo,realData,predictData)
    return realData,predictData

def trainningBatch():
    model =build_model()
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)
    modelcheck_callback = ModelCheckpoint(filepath=weight_file, save_weights_only=True)
    history =model.fit_generator(prepare.trainDataGenerator({"symbol":"000001"}),steps_per_epoch=3,epochs=2,
                                 callbacks=[tensorboard_callback, modelcheck_callback])


def showData(test_y,predict_y):
    x = [i for i in range(len(predict_y))]
    plt.plot(x, predict_y, "b*--")
    plt.plot(x, test_y, "gv--")
    # plt.plot(test_y)
    plt.show()


if __name__ == '__main__':
    trainningAll(only_untrain=True)
    #trainningOne()
    # prepare.dataGenerator()

    #testing({"symbol": "000001"})