import cv2
import numpy as np 

from net import Net128


haar = cv2.CascadeClassifier()
haar.load("./haarcascade_frontalface_default.xml")

base_img = cv2.resize(cv2.imread("./images/user.jpg"),(128,128))
net = Net128(base_img,"./lfw/128_by_128_emb.h5")

def main():
    cap = cv2.VideoCapture(0)

    while True:
        _,frame = cap.read()
        frame = cv2.medianBlur(frame,3)
        gray  = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        faces = haar.detectMultiScale(gray)

        for x,y,w,h in faces:
            face = cv2.resize(frame[y:y+h,x:x+w],(128,128))
            score = net(face)
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)
            frame = cv2.putText(frame,str(score),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))

        cv2.imshow("FRAME",frame)
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()