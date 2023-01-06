import cv2
import numpy as np
from glcm_algorithm import calculate_glcm
from knn_process import knn

img = cv2.imread("knn/dataset/data/skin/melanoma/ISIC_0000141.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


features = calculate_glcm(gray)
print("Properties adalah ", features)

features = np.array(features)
features = features.reshape(1, -1)

print("Properties hasil reshape adalah ", features)

pred = knn.predict(features)

print("Hasil prediksi adalah ", pred)

#%%
