import tensorflow as tf

K = tf.keras.backend

class RMSEScore(tf.keras.layers.Layer):
    def __init__(self, base, **kwargs):
        self.base = tf.constant(base)
        super(RMSEScore, self).__init__(**kwargs)
    def build(self, input_shape):
        super(RMSEScore, self).build(input_shape)  
    def call(self, x):
        return K.sqrt(K.mean(K.square(tf.subtract(self.base,x)),axis=1))
    def compute_output_shape(self, input_shape):
        return [1]

class Net128:
    def __init__(self,base,weights_path):
        self.inputs = tf.keras.layers.Input(shape=(128,128,3))
        self.conv1 = tf.keras.layers.Conv2D(16,4,activation="relu",input_shape=(128,128,3))(self.inputs)
        self.conv2 = tf.keras.layers.Conv2D(16,2,activation="relu",)(self.conv1)
        self.pool1 = tf.keras.layers.MaxPool2D()(self.conv2)
        self.conv3 = tf.keras.layers.Conv2D(32,3,activation="relu",)(self.pool1)
        self.conv4 = tf.keras.layers.Conv2D(32,3,activation="relu",)(self.conv3)
        self.pool2 = tf.keras.layers.MaxPool2D()(self.conv4)
        self.flat = tf.keras.layers.Flatten()(self.pool2)
        self.out = tf.keras.layers.Dense(128)(self.flat)

        self.embedding_layer = tf.keras.models.Model(self.inputs,self.out)
        self.embedding_layer.load_weights(weights_path)
        
        self.base = self.embedding_layer.predict(base.reshape(1,128,128,3))
        self.scoring_layer = RMSEScore(base=self.base)(self.out)
        self.model = tf.keras.models.Model(self.inputs,self.scoring_layer)   
        
    def __call__(self,inputs):
        assert len(inputs.shape) > 2,"Input Must Have Dimentions Greater Then 2"
        return self.model.predict(inputs.reshape(-1,128,128,3))


class Net64:
    def __init__(self,base,weights_path):
        self.inputs =  tf.keras.layers.Input(shape=(64,64,3))
        self.conv1 =  tf.keras.layers.Conv2D(16,4,activation="relu",padding="same",input_shape=(64,64,3))(self.inputs)
        self.conv2 =  tf.keras.layers.Conv2D(16,2,activation="relu",)(self.conv1)
        self.pool1 =  tf.keras.layers.MaxPool2D()(self.conv2)
        self.conv3 =  tf.keras.layers.Conv2D(32,3,activation="relu",)(self.pool1)
        self.conv4 =  tf.keras.layers.Conv2D(32,3,activation="relu",)(self.conv3)
        self.pool2 =  tf.keras.layers.MaxPool2D()(self.conv4)
        self.flat =  tf.keras.layers.Flatten()(self.pool2)
        self.out =  tf.keras.layers.Dense(128)(self.flat)

        self.embedding_layer =  tf.keras.models.Model(self.inputs,self.out)
        self.embedding_layer.load_weights(weights_path)
        
        self.base = self.embedding_layer.predict(base.reshape(1,64,64,3))
        self.scoring_layer = RMSEScore(base=self.base)(self.out)
        self.model =  tf.keras.models.Model(self.inputs,self.scoring_layer)   
        
    def __call__(self,inputs):
        assert len(inputs.shape) > 2,"Input Must Have Dimentions Greater Then 2"
        return self.model.predict(inputs.reshape(-1,64,64,3))