import pickle
import face_recognition
import os

model = pickle.load(open("encodings.pkl", "rb"))
load = os.listdir("unknown_fa")
known_faces = []
unknown_faces = []
for images in load:
    lo = face_recognition.load_image_file("unknown_fa/" + images)
    encodings = face_recognition.face_encodings(lo)
    if len(encodings) > 0:
        encoding = encodings[0]
        match = False
        name = None
        for mod in model:
            if face_recognition.compare_faces([mod["encoding"]], encoding, tolerance=0.6)[0]:
                match = True
                name = mod["name"]
                break
        if match:
            known_faces.append((images, name))
        else:
            unknown_faces.append(images)
for face, name in known_faces:
    print(f"Known {face} , {name}")
for image in unknown_faces:
    print(f"Unknown{image}")