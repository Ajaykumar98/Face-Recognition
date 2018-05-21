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

index = {}
radius=8
numPoints=24

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)

def describe(image, eps=1e-7):
		lbp = feature.local_binary_pattern(image,numPoints, radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(), bins=range(0, numPoints + 3), range=(0, numPoints + 2))
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
		return hist

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

lbph=[]
for i in range(0,len(traindata)):
    image=traindata[i]
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = describe(gray)
    lbph.append(hist)

model = model =SVC(kernel="poly", degree=11, coef0=11)
model.fit(lbph, trainlabel)

lbph=[]
for i in range(0,len(testdata)):
    image=testdata[i]
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = describe(gray)
    lbph.append(hist)
print(classification_report(testlabel, model.predict(lbph)))

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
        hist = describe(roi_gray)
        prediction = model.predict(hist.reshape(1, -1))[0]
        cv2.putText(img, prediction, (x+70, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 4)

    cv2.imshow('img', cv2.resize(img,(1280,720)))
    if (cv2.waitKey(1) == 27):
        break

cap.release()
cv2.destroyAllWindows()
