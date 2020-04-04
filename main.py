import cv2
import numpy as np 

from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import Sequential
from tensorflow.compat.v1.keras.layers import  Input,Conv2D,Flatten,Dense
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras import backend as K


def getModel():
    inputs = Input(shape=(64,64,3))
    conv = Conv2D(8,4,activation="relu",input_shape=(64,64,3))(inputs)
    conv = Conv2D(16,3,activation="relu",)(conv)
    conv = Conv2D(32,2,activation="relu",)(conv)
    flat = Flatten()(conv)
    dense = Dense(128,activation="relu")(flat)
    out = Dense(128)(dense)
    embedding = Model(inputs,out)
    embedding.load_weights("./lfw/emb.h5")
    return embedding

haar = cv2.CascadeClassifier()
haar.load("./haarcascade_frontalface_default.xml")
base_img = cv2.imread("./images/user.jpg")
x,y,w,h = haar.detectMultiScale(base_img.mean(axis=2).astype(np.uint8))[0]
base_img = cv2.resize(base_img[y:y+h,x:x+w],(64,64)).reshape(1,64,64,3)

model = getModel()
base_enb = model.predict(base_img)

def score(values): 
    return np.mean(np.square(base_enb - values))

def main():
    cap = cv2.VideoCapture(0)

    while True:
        _,frame = cap.read()
        gray  = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        faces = haar.detectMultiScale(gray)

        for x,y,w,h in faces:
            face = cv2.resize(frame[y:y+h,x:x+w],(64,64)).reshape(1,64,64,3)
            embeddings = model.predict(face)
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)
            frame = cv2.putText(frame,str(score(embeddings)),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))

        cv2.imshow("FRAME",frame)
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()