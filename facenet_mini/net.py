import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import *

def base_network(d=d):
    _input = Input(shape=(110,110,3))

    a_conv0 = Conv2D(32,3,activation="relu",padding="same")(_input)
    a_conv1 = Conv2D(32,3,activation="relu",padding="same")(a_conv0)
    a_conv2 = Conv2D(32,3,activation="relu",padding="same")(a_conv1)
    a_conc = concatenate([a_conv0,a_conv2,a_conv2])
    a_pool = MaxPool2D()(a_conc)

    b_conv0 = Conv2D(32,3,activation="relu",padding="same")(a_pool)
    b_conv1 = Conv2D(32,3,activation="relu",padding="same")(b_conv0)
    b_conv2 = Conv2D(32,3,activation="relu",padding="same")(b_conv1)
    b_conc = concatenate([a_pool,b_conv0,b_conv2,b_conv2])
    b_pool = MaxPool2D()(b_conc)

    c_conv0 = Conv2D(64,3,activation="relu",padding="same")(b_pool)
    c_conv1 = Conv2D(64,3,activation="relu",padding="same")(c_conv0)
    c_conv2 = Conv2D(64,3,activation="relu",padding="same")(c_conv1)
    c_conc = concatenate([b_pool,c_conv0,c_conv2,c_conv2])
    c_pool = MaxPool2D()(c_conc)

    d_conv0 = Conv2D(64,3,activation="relu",padding="same")(c_pool)
    d_conv1 = Conv2D(64,3,activation="relu",padding="same")(d_conv0)
    d_conv2 = Conv2D(64,3,activation="relu",padding="same")(d_conv1)
    d_conc = concatenate([c_pool,d_conv0,d_conv2,d_conv2])
    d_pool = MaxPool2D()(d_conc)

    e_conv0 = Conv2D(128,3,activation="relu",padding="same")(d_pool)
    e_conv1 = Conv2D(128,3,activation="relu",padding="same")(e_conv0)
    e_conv2 = Conv2D(128,3,activation="relu",padding="same")(e_conv1)
    e_conv3 = Conv2D(128,3,activation="relu",padding="same")(e_conv2)
    e_conc = concatenate([d_pool,e_conv0,e_conv1,e_conv2,e_conv3])
    e_pool = MaxPool2D()(e_conc)

    f_conv0 = Conv2D(256,3,activation="relu",padding="same")(e_pool)
    f_conv1 = Conv2D(256,3,activation="relu",padding="same")(f_conv0)
    f_conv2 = Conv2D(256,3,activation="relu",padding="same")(f_conv1)
    f_conv3 = Conv2D(256,3,activation="relu",padding="same")(f_conv2)
    f_conc = concatenate([f_conv0,f_conv1,f_conv2,f_conv3])
    f_pool = MaxPool2D()(f_conc)

    dense = Flatten()(f_pool)
    dense = Dense(512,activation="linear")(dense)
    dense = Dense(d,activation="tanh")(dense)
    dense = tf.multiply(dense,128)
    model = keras.Model(_input,dense,name="FaceNet")

    return model