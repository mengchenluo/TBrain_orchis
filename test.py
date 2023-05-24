from keras.models import load_model
import numpy as np
import cv2, os
import pandas as pd
from time import time

filename = []
predict = []
model = load_model("osme_resnet50_Adam_encolor.hdf5")
dir1 = os.listdir("orchid_public_set")
dir2 = os.listdir("orchid_private_set")
start = time()
for name in dir1:
        print(name)
        img = cv2.imread("orchid_public_set/"+name, 1)
        if img.shape!=(224, 224, 3):
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = img / 255.0
        img = np.array([img])
        cls = np.argmax(model.predict(img), axis=1)
        filename.append(name)
        predict.append(cls[0])
for name in dir2:
        print(name)
        img = cv2.imread("orchid_private_set/"+name, 1)
        if img.shape!=(224, 224, 3):
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = img / 255.0
        img = np.array([img])
        cls = np.argmax(model.predict(img), axis=1)
        filename.append(name)
        predict.append(cls[0])
end = time()
print("Time:", end-start)
submission = pd.DataFrame({
        "filename": filename,
        "category": predict
})
submission.to_csv("submission.csv", index=False)