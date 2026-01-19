import cv2
import os
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []

dataset_path = "dataset"

for label in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, label)
    if not os.path.isdir(person_path):
        continue

    for file in os.listdir(person_path):
        img_path = os.path.join(person_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        faces.append(img)
        labels.append(int(label))

recognizer.train(faces, np.array(labels))
recognizer.save("face_model.yml")

print("Model trained successfully!")
