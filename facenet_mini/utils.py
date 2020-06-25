import numpy as np
import cv2

from tqdm.cli import tqdm
from os import walk,path as pathlib
from glob import glob

class Pairs(object):
    def __init__(self,model, x:np.ndarray,y:np.ndarray,image_shape=(110,110,3)):
        assert len(x) == len(y)
        self.x = x.reshape(-1,*x[0].shape[:2],3)
        self.y = y
        self.model = model
        self.dummy = np.array([[0]])
        self.image_shape = image_shape
        
    def get_pair(self,e,x,y):
        a = x.reshape(1,*self.image_shape)
        
        p_index = np.where(self.y == y)
        n_index = np.where(self.y != y)

        p = self.epoch_enc[p_index]
        n = self.epoch_enc[n_index]

        p_dist = np.sum(np.square(p - e),axis=1).argmax()
        n_dist = np.sum(np.square(n - e),axis=1).argmin()

        p = self.x[p_index][p_dist].reshape(1,*self.image_shape)
        n = self.x[n_index][n_dist].reshape(1,*self.image_shape)
        
        return np.array([a,p,n])
        
    def flow(self,epochs=1):
        for epoch in range(epochs):
            self.epoch_enc = self.model.predict(self.x,batch_size=600)
            _iter = tqdm(zip(self.epoch_enc,self.x,self.y),total=len(self.epoch_enc))
            this_batch = np.array([self.get_pair(e,x,y) for e,x,y in _iter])
            for a,p,n in this_batch:
                yield (a,p,n),self.dummy

        



class Dataset(object):
    """
    Dataset wrapper to read images from folder and structure it.
    """
    def __init__(self,path,n_faces,resize=110):
        self.path = pathlib.abspath(path)
        self.resize = resize

    def __repr__(self,):
        return f"Dataset Parser @ {self.path}"
        
    def parse(self,progress_bar=True):
        w = walk(self.path)
        self.data = np.array([[[r,pathlib.join(r,f_)] for f_ in f] for r,_,f in list(w)[1:]]).reshape(-1,2)
        self.y = np.array(self.data[:,0])
        self.x = np.array([
            cv2.cvtColor(
                cv2.resize(
                    cv2.imread(
                        i
                    ),
                    (self.resize,self.resize)
                ),
                cv2.COLOR_BGR2RGB
            ) / 255
            for 
                i
            in 
                (tqdm(self.data[:,1]) if progress_bar else self.data[:,1])
        ]).astype(np.float32)
        return self
            
















