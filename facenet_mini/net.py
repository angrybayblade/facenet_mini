import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *


def block(_input:Layer,filters:int,name:str,ksize:int=3):
    conv0 = Conv2D(filters,ksize,activation="relu",padding="same",name=f"conv_0_{name}")(_input)
    conv1 = Conv2D(filters,ksize,activation="relu",padding="same",name=f"conv_1_{name}")(conv0)
    conv2 = Conv2D(filters,ksize,activation="relu",padding="same",name=f"conv_2_{name}")(conv1)
    conc  = concatenate([conv0,conv2,conv2],name=f"conc_{name}")
    pool  = MaxPool2D(name=f"pool_{name}")(conc)

    return pool


def base_network(vector_length:int=128,input_shape:tuple=(110,110,3),multiplication_factor = 32):
    """
    :param d : dimension of embedding vector
    :param input_shape: shape of input vector
    :param multiplication_factor: value of multiplication factor which will decide 
                                range of values for embedding space.
    """
    _input = Input(shape=input_shape)

    a = Conv2D(16,7,activation="relu",padding="same",strides=2)(_input)
    b = block(a,32,"b",3)
    c = block(b,64,"c",3)
    d = block(c,128,"d",3)
    e = block(d,512,"e",3)
    
    f = GlobalMaxPool2D()(e)

#     dense = Flatten()(f)
    dense = Dense(1024,activation="tanh")(f)
    dense = Dense(vector_length,activation="tanh")(dense)

    dense = tf.multiply(dense,multiplication_factor)
    model = keras.Model(_input,dense)

    return model

def builder(model:tf.keras.Model=None,vector_length:int=128,input_shape:tuple=(110,110,3),multiplication_factor:int=32):
    if not model:
        model = base_network(vector_length,input_shape,multiplication_factor)

    a_inp = Input(shape=input_shape,name="anc")
    p_inp = Input(shape=input_shape,name="pos")
    n_inp = Input(shape=input_shape,name="neg")

    a_net = model(a_inp)
    p_net = model(p_inp)
    n_net = model(n_inp)

    out = concatenate([a_net,p_net,n_net],name="out")
    return  keras.Model([a_inp,p_inp,n_inp],out,name="triplet_trainig")



class Triplet(tf.Module):
    """
    Triplet Loss
    """
    __name__ = "TripletLoss"
    def __init__(self,margin=.75,vector_length:int=128):
        self.margin = margin
        self.vector_length = vector_length
        
    @tf.function
    def l2(self,x,y):
        return tf.reduce_sum(tf.square(tf.subtract(x,y)))
    
    @tf.function
    def __call__(self,y_true,y_pred,*args,**kwargs):
        a,p,n = tf.unstack(tf.reshape(y_pred,(3,-1,self.vector_length)))
        
        Dp = self.l2(a,p)
        Dn = self.l2(a,n)
        
        return tf.nn.relu(Dp - Dn + self.margin)


class StopTraining(keras.callbacks.Callback):
    def __init__(self,threshold=1e-3):
        self.loss_history = [*list(range(5))]
        self.threshold = threshold
        
    def on_epoch_end(self,epoch,logs=dict()):
        epoch_loss = logs.get('loss')
        self.loss_history.append(epoch_loss)
        if (sum(self.loss_history[-5:])/5) < self.threshold:
            self.model.stop_training = True 
            print (f"Stopped Training At {epoch} Epochs.")