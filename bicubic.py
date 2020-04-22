import cv2
import tensorflow as tf

tf.enable_eager_execution()

cap = cv2.VideoCapture(0)
while True:
    _,frame = cap.read()
    cv2.imshow("original",frame)
    cv2.imshow("CV RESIZED",cv2.resize(frame,(1000,640)))
    frame = tf.cast(tf.image.resize_bicubic(frame.reshape(-1,*frame.shape),(640,1000)),tf.uint8).numpy()[0]
    cv2.imshow("TF RESIZED",cv2.medianBlur(frame,3))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

