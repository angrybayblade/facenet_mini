import cv2

haar = cv2.CascadeClassifier()
haar.load("./haarcascade_frontalface_default.xml")


def main():
    cap = cv2.VideoCapture(0)

    while True:
        _,frame = cap.read()
        img = frame.copy()
        gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        faces = haar.detectMultiScale(gray)

        for x,y,w,h in faces:
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
        cv2.imshow("faces",frame)
        key = cv2.waitKey(1) & 0xFF


        if key  == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("./images/user.jpg",img)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()