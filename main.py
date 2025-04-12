import face_recognition
import os
import numpy
import pickle

img = os.listdir("known_faces")
list = []
for images in img:
    load = face_recognition.load_image_file("known_faces/" + images)
    encodings = face_recognition.face_encodings(load)[0]
    face_location = face_recognition.face_locations(load)[0]
    face_landmark = face_recognition.face_landmarks(load)[0]
    images = images.split(".")[0]
    list.append({"name":images,"encoding":encodings.tolist(),"location":face_location,"landmark":face_landmark})

pickle.dump(list, open("encodings.pkl", "wb"))
print(list)