import cv2
import random
import winsound
import _thread


trained_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0)


def playsound():
    winsound.Beep(500, 200)


while True:
    success_read, frame1 = webcam.read()
    gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    face_coordinates = trained_face.detectMultiScale(gray)
    print(face_coordinates)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame1, (x, y), (x + w, y + h),(random.randrange(128, 256), random.randrange(128, 256), random.randrange(128, 256)), 2)
        if len(face_coordinates) > 0:
            _thread.start_new_thread(playsound, ())
    if cv2.waitKey(10) == ord('q'):
        break
    cv2.imshow('Face Detector', frame1)




