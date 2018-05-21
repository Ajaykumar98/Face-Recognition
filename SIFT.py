from imutils import paths
import numpy as np
import argparse
import cv2
from skimage import feature
from sklearn.metrics import classification_report
from sklearn.svm import SVC

#--train Dataset1\TrainImage --test Dataset1\TestImage

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True, help="path to the training imag")
ap.add_argument("--test", required=True, help="path to the testing image")
args = vars(ap.parse_args())


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)

trainlabel=[]
trainlabelunique=[]
for imagePath in paths.list_files(args["train"]):
    if imagePath.split("\\")[-2] not in trainlabelunique:
        trainlabelunique.append(imagePath.split("\\")[-2])
    trainlabel.append(imagePath.split("\\")[-2])

traindata=[]
for i in range (0,len(trainlabelunique)):
    for imagePath in paths.list_images(args["train"]+'\\'+trainlabelunique[i]):
        traindata.append(cv2.imread(imagePath))

testlabel=[]
testlabelunique=[]
for imagePath in paths.list_files(args["test"]):
    if imagePath.split("\\")[-2] not in testlabelunique:
	    testlabelunique.append(imagePath.split("\\")[-2])
    testlabel.append(imagePath.split("\\")[-2])

testdata=[]
for i in range (0,len(testlabelunique)):
    for imagePath in paths.list_images(args["test"]+'\\'+testlabelunique[i]):
        testdata.append(cv2.imread(imagePath))

sift=[]
for i in range(0,len(traindata)):
    image=traindata[i]
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    SIFT=cv2.xfeatures2d.SIFT_create()
    kp,des=SIFT.detectAndCompute(gray,None)
    sift_feature=[]
   # print des.shape
    for j in range(0,des.shape[1]):
        sift_feature.append(0.0)
    for j in range(0,des.shape[0]):
        for k in range(0,des.shape[1]):
            sift_feature[k]+=des[j][k]
    for j in range(0, des.shape[1]):
        sift_feature[j]/=des.shape[0]
    sift.append(sift_feature)

model =SVC(kernel="poly", degree=5, coef0=-15)
model.fit(sift, trainlabel)

sift=[]
for i in range(0,len(testdata)):
    image=testdata[i]
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    SIFT = cv2.xfeatures2d.SIFT_create()
    kp, des = SIFT.detectAndCompute(gray, None)
    sift_feature = []
    for j in range(0, des.shape[1]):
        sift_feature.append(0.0)
    for j in range(0, des.shape[0]):
        for k in range(0, des.shape[1]):
            sift_feature[k] += des[j][k]
    for j in range(0, des.shape[1]):
        sift_feature[j] /= des.shape[0]
    sift.append(sift_feature)

print(classification_report(testlabel, model.predict(sift)))
dim=(100,100)
while True:

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, dim, interpolation=cv2.INTER_AREA)
        SIFT = cv2.xfeatures2d.SIFT_create()
        kp, des = SIFT.detectAndCompute(roi_gray, None)
        sift_feature = []
        for j in range(0, des.shape[1]):
            sift_feature.append(0.0)
        for j in range(0, des.shape[0]):
            for k in range(0, des.shape[1]):
                sift_feature[k] += des[j][k]
        for j in range(0, des.shape[1]):
            sift_feature[j] /= des.shape[0]
        prediction = model.predict(np.asarray(sift_feature).reshape(1, -1))[0]
        cv2.putText(img, prediction, (x+70, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255, 0), 4)

    cv2.imshow('img', cv2.resize(img,(1280,720)))
    if (cv2.waitKey(1) == 27):
        break

cap.release()
cv2.destroyAllWindows()
