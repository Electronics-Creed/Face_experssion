import numpy as np
import cv2 as cv
from keras.models import model_from_json
from keras_preprocessing import image

model = model_from_json(open('fer.json', 'r').read())
model.load_weights('fer.h5')

face_haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, test_img = cap.read()
    if not ret:
        continue
    gray_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)  # get the faces in img/frame

    for x, y, w, h in faces_detected:
        cv.rectangle(test_img, (x, y), (x + w, y + h), (0, 225, 225), 5)  # draw rectangle for face
        roi_gray = gray_img[y:y + h, x:x + h]  # cut face
        roi_gray = cv.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 225

        preditions = model.predict(img_pixels)

        max_index = np.argmax(preditions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotions = emotions[max_index]

        cv.putText(test_img, predicted_emotions, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1, (225, 225, 0), 2,
                   cv.LINE_AA)

    resized_img = cv.resize(test_img, (1000, 700))
    cv.imshow('Facial emotion analysis', resized_img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
