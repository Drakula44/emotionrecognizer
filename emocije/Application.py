import cv2
import math
import numpy as np
import dlib
import pickle

emotions = ["neutralno", "tuzno", "srecno", "uplaseno", "ljuto", "uzasnuto", "iznenadjeno"]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
filename = "finalmodel"
loaded_model = pickle.load(open(filename, 'rb'))

data = {}
l = 0

def get_landmarks(detections, image):
    for k, d in enumerate(detections):
        shape = predictor(image, d)
        xlist = []
        ylist = []
        for i in range(1, 68):
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]
        ycentral = [(y - ymean) for y in ylist]

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append(int(math.atan((y - ymean) / (x - xmean)) * 360 / math.pi))

        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"

def crop_face(gray, face): #Crop the given face
    out = []
    l = []
    for (x, y, w, h) in face:
        out.append(cv2.resize(gray[y:y+h, x:x+w], (350,350)))
        l.append(w)
    return (out, l)

video_capture = cv2.VideoCapture(0)
width = 125
height = 20
br = 0
max = 0
while True:
    ret, frame = video_capture.read() #capture frame, ret==true if captured correctly
    if True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to gray scale
        clahe_image = clahe.apply(gray)

        #detect face on image
        face = facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(face) == 0:
            cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("image", frame)
        if len(face) > 0:
            (faceslice, l) = crop_face(gray, face)  # slice face from image
            j = 0
            for faces in faceslice:
                detections = detector(faces, 1)
                if br == 0:
                    get_landmarks(detections, faces)
                    test = np.array(data['landmarks_vectorised'])
                    predictions = loaded_model.predict_proba(test.reshape(1, -1))
                    for num in predictions[0]:
                        if num > max:
                            max = num
                for i in range(len(emotions)):
                    if(max == predictions[0][i]):
                        cv2.rectangle(frame, ((170 + 200*j,20 + 25*i)), ((170 + 200*j + int(width*predictions[0][i]),20 + 25*i + height)), (0,255,0), thickness=cv2.FILLED)
                    else:
                        cv2.rectangle(frame, ((170 + 200*j,20 + 25*i)), ((170 + 200*j + int(width*predictions[0][i]),20 + 25*i + height)), (105,105,105), thickness=cv2.FILLED)
                    cv2.putText(frame, "%s: %.2f" % (emotions[i], predictions[0][i]*100),(20 + 200*j,30 + 25*i), cv2.FONT_HERSHEY_SIMPLEX, thickness=2, fontScale=0.5, color=(0,0,0))
                    #frame[10 + 60*j:10 + 60*j, 500:50] = cv2.resize(images[i], None, fx=50/images[i].shape()[1], fy=50/images[i].shape()[0], interpolation=cv2.INTER_CUBIC)
                for k, d in enumerate(detections):
                    shape = predictor(faces, d)
                    for i in range(1, 68):
                        x = int(shape.part(i).x * l[j] / 350) + face[j][0]
                        y = int(shape.part(i).y * l[j] / 350) + face[j][1]
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), thickness=2)
                j+= 1
            cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("image", frame)
    br += 1
    if br == 2:
        br = 0
        max = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

