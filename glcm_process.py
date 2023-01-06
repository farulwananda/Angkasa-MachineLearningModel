import numpy as np
import cv2
import os
import re
from skimage.feature import greycomatrix, greycoprops
import pandas as pd


# -------------------- Load Dataset ------------------------

def normalize_label(str_):
    str_ = str_.replace(" ", "")
    str_ = str_.translate(str_.maketrans("", "", "()"))
    str_ = str_.split("_")
    return ''.join(str_[:2])


def normalize_desc(folder, sub_folder):
    text = folder + " - " + sub_folder
    text = re.sub(r'\d+', '', text)
    text = text.replace(".", "")
    text = text.strip()
    return text


def print_progress(val, val_len, folder, sub_folder, filename, bar_size=10):
    progr = "#" * round((val) * bar_size / val_len) + " " * round((val_len - (val)) * bar_size / val_len)
    if val == 0:
        print("", end="\n")
    else:
        print("[%s] folder : %s/%s/ ----> file : %s" % (progr, folder, sub_folder, filename), end="\r")


dataset_dir = "knn/dataset/data"

imgs = []  # list image matrix
labels = []
descs = []
for folder in os.listdir(dataset_dir):
    for sub_folder in os.listdir(os.path.join(dataset_dir, folder)):
        sub_folder_files = os.listdir(os.path.join(dataset_dir, folder, sub_folder))
        len_sub_folder = len(sub_folder_files) - 1
        for i, filename in enumerate(sub_folder_files):
            img = cv2.imread(os.path.join(dataset_dir, folder, sub_folder, filename))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            h, w = gray.shape
            ymin, ymax, xmin, xmax = h // 3, h * 2 // 3, w // 3, w * 2 // 3
            crop = gray[ymin:ymax, xmin:xmax]

            resize = cv2.resize(crop, (0, 0), fx=0.5, fy=0.5)
            imgs.append(resize)
            labels.append(sub_folder)
            descs.append(normalize_desc(folder, sub_folder))


# ----------------- calculate greycomatrix() & greycoprops() for angle 0, 45, 90, 135 ----------------------------------

def calc_glcm_all_agls(img, label, props, dists=[5], agls=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], lvl=256, sym=True,
                       norm=True):
    glcm = greycomatrix(img,
                        distances=dists,
                        angles=agls,
                        levels=lvl,
                        symmetric=sym,
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[0]]
    for item in glcm_props:
        feature.append(item)
    feature.append(label)

    return feature


# ----------------- call calc_glcm_all_agls() for all properties ----------------------------------
properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

glcm_all_agls = []
for img, label in zip(imgs, labels):
    glcm_all_agls.append(
        calc_glcm_all_agls(img,
                           label,
                           props=properties)
    )

columns = []
angles = ['0', '45', '90', '135']
for name in properties:
    for ang in angles:
        columns.append(name + "_" + ang)

columns.append("label")

glcm_df = pd.DataFrame(glcm_all_agls,
                       columns=columns)

glcm_df.to_csv('glcmx6.csv')
