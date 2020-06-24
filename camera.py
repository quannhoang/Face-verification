import cv2
import os
from model import FacePredictModel
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("VGG_model_test.json", "VGG_model_test_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
epsilon = 0.2 #cosine similarity

def preprocess_img(path):
    img = image.load_img(path, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img)
    return img

def preprocess_img_from_opencv(img):
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img)
    return img

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def verifyFace(img1_representation, img2_representation):
 #img1_representation = face_to_vector.predict(img1)[0,:]
 #img2_representation = face_to_vector.predict(img2)[0,:]
 
 cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
 #euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
 
 if(cosine_similarity < epsilon):
  return True
 else:
  return False

face_db = {}
for person in os.listdir('face_db'):
    person_vectors = np.zeros([2622,], dtype='float64')
    count = 0
    for filename in os.listdir('face_db/'+ person):
        count += 1
        single_vector = model.predict_emotion(preprocess_img('face_db' + '/' + person + '/' + filename))[0,:]
        person_vectors += single_vector
    person_vectors /= count
    #person_vectors = np.asarray(person_vectors)
    face_db[person] = person_vectors
    
class VideoCamera(object):
    def __init__(self):
        #self.video = cv2.VideoCapture(0)
        #self.video = cv2.VideoCapture('E:/anaconda3/1 Projects Folder/Facial expression recognition/videos/facial_exp.mkv')
        self.video = cv2.VideoCapture('E:/anaconda3/1 Projects Folder/Face Verification/videos/Presidential debate highlights Clinton and Trump trade blows – video.mp4')

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        #gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (224, 224))
            
            pred = model.predict_face(preprocess_img_from_opencv(roi))
            pred = pred.reshape(2622,)
            if verifyFace(pred, face_db['Trump']):
                cv2.putText(fr, 'Trump', (x, y), font, 1, (255, 255, 0), 2)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
            else:
                cv2.putText(fr, 'Stranger', (x, y), font, 1, (255, 255, 0), 2)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)                      

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
