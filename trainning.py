import sys
# print(sys.executable)
import numpy as np
import matplotlib.pyplot as plt
import gc

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import datetime
import tensorflow.keras as keras
import prepare

import stock_sql
from abc import ABCMeta, abstractmethod
import time

import pickle

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
# print(gpus)
# print(cpus)
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
# tf.config.experimental.set_visible_devices(devices=cpus[0], device_type='CPU')
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[1],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1536)]
# )


weight_file = "".join(["model/weights/my_weight"])
weight_file_1 = "".join(["model/weights/my_weight_1"])
index_file = "index.txt"
log_dir = "".join(
    ["log\\model_train\\", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")])
model = None


def getTrainIndex():
    with open(index_file, "r") as f:
        index = f.readline()
    return int(index)


def saveTrainIndex(index):
    with open(index_file, "w") as f:
        f.write(str(index))
        f.flush()


def build_model(shape):
    model = model_1(shape)
    keras.utils.plot_model(model, 'picture/multi_model.png', show_shapes=True)
    return model


def model_2(shape):
    model = keras.Sequential()
    model.add(keras.Input(shape=shape))
    model.add(keras.layers.LSTM(
        units=50, activation='tanh', return_sequences=True))
    model.add(keras.layers.LSTM(
        units=50, activation='tanh', return_sequences=True))
    model.add(keras.layers.Dense(units=10, activation="tanh"))
    model.add(keras.layers.Dense(units=1))
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss="mse",
                  metrics=[keras.metrics.mae])
    model.summary()
    return model

def testing(stockInfo):
    """

    :param stockInfo:{symbol: XXX,name}
    :return:
    """
    return logisticsTrain().predict(stockInfo)

def showData(test_y, predict_y):
    x = [i for i in range(len(predict_y))]
    plt.plot(x, predict_y, "b*--")
    plt.plot(x, test_y, "gv--")
    # plt.plot(test_y)
    plt.show()


def template():
    # 构建一个根据文档内容、标签和标题，预测文档优先级和执行部门的网络
    # 超参
    num_words = 2000
    num_tags = 12
    num_departments = 4

    # 输入
    body_input = keras.Input(shape=(None,), name='body')
    title_input = keras.Input(shape=(None,), name='title')
    tag_input = keras.Input(shape=(num_tags,), name='tag')

    # 嵌入层
    body_feat = keras.layers.Embedding(num_words, 64)(body_input)
    title_feat = keras.layers.Embedding(num_words, 64)(title_input)

    # 特征提取层
    body_feat = keras.layers.LSTM(32)(body_feat)
    title_feat = keras.layers.LSTM(128)(title_feat)
    features = keras.layers.concatenate([title_feat, body_feat, tag_input])

    # 分类层
    priority_pred = keras.layers.Dense(
        1, activation='sigmoid', name='priority')(features)
    department_pred = keras.layers.Dense(
        num_departments, activation='softmax', name='department')(features)

    # 构建模型
    model = keras.Model(inputs=[body_input, title_input, tag_input],
                        outputs=[priority_pred, department_pred])
    # model.summary()
    keras.utils.plot_model(
        model, 'picture/template_model.png', show_shapes=True)


def logisticsMode(shape):
    model = keras.Sequential()
    model.add(keras.Input(shape=shape))
    #model.add(keras.layers.Dense(units=100, activation="tanh"))
    model.add(keras.layers.LSTM(units=500, activation='tanh', return_sequences=True))
    model.add(keras.layers.LSTM(units=500, activation='tanh', return_sequences=True))
    model.add(keras.layers.LSTM(units=200, activation='tanh', return_sequences=True))
    model.add(keras.layers.LSTM(units=200, activation='tanh', return_sequences=False))
    model.add(keras.layers.Dense(units=200, activation="tanh"))
    model.add(keras.layers.Dense(units=20, activation="tanh"))
    model.add(keras.layers.Dense(units=1))
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss="mse",
                  metrics=[keras.metrics.mae])
    model.summary()
    return model

def classifyModel(shape):

    inn = keras.Input(shape=shape)
    lstm1 = keras.layers.LSTM(units=500, activation='tanh', return_sequences=True)(inn)
    lstm2 = keras.layers.LSTM(units=500, activation='tanh', return_sequences=True)(lstm1)
    lstm3 = keras.layers.LSTM(units=200, activation='tanh', return_sequences=True)(lstm2)
    lstm4 = keras.layers.LSTM(units=50, activation='tanh', return_sequences=True)(lstm3)
    flatten = keras.layers.Flatten()(lstm4)
    Dense1 = keras.layers.Dense(units=200, activation="relu")(flatten)
    ott = keras.layers.Dense(units=3)(Dense1)

    model = keras.Model(inputs=inn, outputs=ott)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    # keras.utils.plot_model(model,"picture/classify_model.png",show_shapes=True)
    return model


def classifyModel_1(shape):
    inn = keras.Input(shape=shape)
    lstm1 = keras.layers.LSTM(units=500, activation='tanh', return_sequences=True)(inn)
    lstm2 = keras.layers.LSTM(units=500, activation='tanh', return_sequences=True)(lstm1)
    lstm3 = keras.layers.LSTM(units=200, activation='tanh', return_sequences=True)(lstm2)
    lstm4 = keras.layers.LSTM(units=200, activation='tanh', return_sequences=True)(lstm3)
    lstm5 = keras.layers.LSTM(units=100, activation='tanh', return_sequences=True)(lstm4)
    flatten = keras.layers.Flatten()(lstm5)
    Dense1 = keras.layers.Dense(units=200, activation="relu")(flatten)
    Dense2 = keras.layers.Dense(units=100,activation="relu")(Dense1)
    ott =  keras.layers.Dense(units= 3)(Dense2)

    model = keras.Model(inputs=inn, outputs=ott)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    # keras.utils.plot_model(model,"picture/classify_model.png",show_shapes=True)
    return model

def logisticsModel_Ex(daily_shape,area_shape,indu_shape):
    daily_input = keras.Input(shape= daily_shape)
    area_input = keras.Input(shape= area_shape)
    indu_input = keras.Input(shape= indu_shape)

    lstm1 = keras.layers.LSTM(units=500, activation="relu", return_sequences=True)(daily_input)
    lstm2 = keras.layers.LSTM(units=200, activation="relu", return_sequences=True)(lstm1)
    lstm3 = keras.layers.LSTM(units=200, activation="relu", return_sequences=True)(lstm2)
    lstm4 = keras.layers.LSTM(units=200, activation="relu", return_sequences=False)(lstm3)

    # dense_area = keras.layers.Dense(units=200, activation="relu")(area_input)
    # dense_indu = keras.layers.Dense(units=200, activation="relu")(indu_input)

    flatten =keras.layers.concatenate([lstm4,area_input,indu_input])

    dense1 = keras.layers.Dense(units=200, activation="relu")(flatten)
    dense2 = keras.layers.Dense(units=100, activation ="relu")(dense1)
    dense3 = keras.layers.Dense(units=100, activation= "relu")(dense2)
    ott = keras.layers.Dense(units=1)(dense3)

    model =keras.Model(inputs=[daily_input,area_input,indu_input],outputs=ott)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss="mse",
                  metrics=[keras.metrics.mae])
    model.summary()
    return model



class trainBase:
    def __init__(self):
        self.dataPre:prepare.prepareBase =None
        self.model:keras.Model=None
        pass

    @abstractmethod
    def buildModel(self):
        pass

    def trainAll(self):
        for data in self.dataPre.dataGenerator(0):
            train_x, train_y, stock_list = data
            print(train_x[0].shape)
            print(train_x[1].shape)
            print(train_x[2].shape)
            print(train_y.shape)
            history = self.model.fit(train_x, train_y, batch_size=self.batch_size,
                                     epochs=self.epochs, validation_split=0.05)
            self.model.save_weights(self.file_path)
            stock_sql.updateTrianList(stock_list)
            del train_x, train_y, stock_list
            gc.collect()
            gc.collect()

    def predict(self, stockinfo: prepare.stockInfo) -> (np.array, np.array):
        data = stock_sql.getRecordData(stockinfo.symbol)
        if data:
            return data[0], data[1]
        data = self.dataPre.getTestData(stockinfo)
        if data is None:
            return np.array([]), np.array([])
        test_x, real_y,scaler_x, scaler_y = data[0], data[1], data[2], data[3]
        predict_y = self.model.predict(test_x)
        if scaler_y is not  None:
            real_y = scaler_y.inverse_transform(real_y)
            predict_y = scaler_y.inverse_transform(predict_y.astype("float64"))
        real_y, predict_y = np.around( real_y, decimals=2), np.around(predict_y, decimals=2)

        stock_sql.SaveRecordData(stockinfo, real_y, predict_y)
        return real_y, predict_y




class logisticsTrain(trainBase):
    def __init__(self):
        self.file_path = "".join(["model/weights/my_weight"])
        self.batch_size = 1024
        self.epochs = 20
        self.buildModel()

    def buildModel(self):
        print("parepareLogistics")
        self.dataPre = prepare.logisticsAllScaler()

        self.model = logisticsMode(self.dataPre.getInputShape())
        try:
            print("load_weights from " + self.file_path)
            self.model.load_weights(self.file_path)
            print(" success ")
        except:
            print("load_weight failed")
            pass

        time.sleep(10)


class classifyTrain(trainBase):
    def __init__(self):
        self.file_path = "".join(["model/weights/classify_weight"])
        self.batch_size = 1024
        self.epochs = 5
        self.buildModel()

    def buildModel(self):
        self.dataPre = prepare.prepareClassify()
        print("prepareClassify")
        self.model = classifyModel(self.dataPre.getInputShape())
        try:
            print("load_weights from " + self.file_path)
            self.model.load_weights(self.file_path)
            print(" success ")
        except:
            print("load_weight failed")
            pass
        time.sleep(10)

class classifyTrain_1(trainBase):
    def __init__(self):
        self.file_path = "".join(["model/weights/classify_weight_1"])
        self.batch_size = 1024
        self.epochs = 5
        self.buildModel()

    def buildModel(self):
        self.dataPre = prepare.prepareClassify()
        print("prepareClassify")
        self.model = classifyModel_1(self.dataPre.getInputShape())
        try:
            print("load_weights from " + self.file_path)
            self.model.load_weights(self.file_path)
            print(" success ")
        except:
            print("load_weight failed")
            pass
        time.sleep(10)

class logisticsTrain_Ex(trainBase):

    def __init__(self):
        self.file_path ="".join(["model/weights/logisticsEx_weight"])
        self.batch_size = 512
        self.epochs = 5
        self.buildModel()

    def buildModel(self):
        self.dataPre = prepare.prepareLogistics_Ex()
        print("prepareClassifyEx")
        daily_shape, area_shape, indu_shape = self.dataPre.getInputShape()
        self.model = logisticsModel_Ex(daily_shape, area_shape, indu_shape)
        try:
            print("load_weights from " + self.file_path)
            self.model.load_weights(self.file_path)
            print(" success ")
        except:
            print("load_weight failed")
            pass
        time.sleep(10)
        pass

if __name__ == '__main__':
    # stock_sql.initTrainDate()

    train = logisticsTrain_Ex()
    train.dataPre.only_untrain=False
    train.trainAll()
