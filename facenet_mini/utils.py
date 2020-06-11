import numpy as np
import tensorflow as tf

from tqdm.cli import tqdm


class Pairs(object):
    def __init__(self,x,y,model=None):
        assert len(x) == len(y)
        self.x = x.reshape(-1,*x[0].shape[:2],3)
        self.y = y
        self.model = model
        self.dummy = np.array([[0]])
        
    def get_pair(self,e,x,y):
        a = x.reshape(1,110,110,3)
        
        p_index = np.where(self.y == y)
        n_index = np.where(self.y != y)

        p = self.epoch_enc[p_index]
        n = self.epoch_enc[n_index]

        p_dist = np.sum(np.square(p - e),axis=1).argmax()
        n_dist = np.sum(np.square(n - e),axis=1).argmin()

        p = self.x[p_index][p_dist].reshape(1,110,110,3)
        n = self.x[n_index][n_dist].reshape(1,110,110,3)
        
        return np.array([a,p,n])
        
    def flow(self,epochs=1):
        for epoch in range(epochs):
            self.epoch_enc = self.model.predict(self.x,batch_size=600)
            _iter = tqdm(zip(self.epoch_enc,self.x,self.y),total=len(self.epoch_enc))
            this_batch = np.array([self.get_pair(e,x,y) for e,x,y in _iter])
            for a,p,n in this_batch:
                yield (a,p,n),self.dummy

class Triplet(tf.Module):
    """
    Triplet Loss
    """
    def __init__(self,margin=.75):
        self.margin = margin
        
    @tf.function
    def l2(self,x,y):
        return tf.reduce_sum(tf.square(tf.subtract(x,y)))
    
    @tf.function
    def __call__(self,y_true,y_pred,*args,**kwargs):
        a,p,n = tf.unstack(tf.reshape(y_pred,(3,-1,d)))
        
        Dp = self.l2(a,p)
        Dn = self.l2(a,n)
        
        return tf.nn.relu(Dp - Dn + self.margin)
        

class Dataset(object):
    def __init__(
                self,
                path,
                resize,
            ):
        pass
