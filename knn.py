import numpy as np
import cv2
import os
import re
import random
from csv import reader
from math import sqrt
from skimage.feature import greycomatrix, greycoprops
import pandas as pd
from csv import reader
from math import sqrt


# -------------------- Load Dataset ------------------------

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


dataset_dir = "dataset"

imgs = []  # list image matrix
labels = []
descs = []
for folder in os.listdir(dataset_dir):
    for sub_folder in os.listdir(os.path.join(dataset_dir, folder)):
        sub_folder_files = os.listdir(os.path.join(dataset_dir, folder))
        len_sub_folder = len(sub_folder_files) - 1
        for i, filename in enumerate(sub_folder_files):
            img = cv2.imread(os.path.join(dataset_dir, folder, sub_folder, filename))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            h, w = gray.shape
            ymin, ymax, xmin, xmax = h // 3, h * 2 // 3, w // 3, w * 2 // 3
            crop = gray[ymin:ymax, xmin:xmax]

            resize = cv2.resize(crop, (0, 0), fx=0.5, fy=0.5)

            imgs.append(resize)
            labels.append(random.randint(1, 100000))
            descs.append(normalize_desc(folder, sub_folder))

            print_progress(i, len_sub_folder, folder, sub_folder, filename)


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
        columns.append(ang)

columns.append("label")

glcm_df = pd.DataFrame(glcm_all_agls,
                       columns=columns)

glcm_df.to_csv('glcm.csv')
glcm_df.head(15)


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    i = 0
    try:
        for row in dataset:
            row[column] = float(row[column].strip())
    except ValueError:
        print(f'Change value: {row[column]} on row {i} column {column} to numeric.')
    finally:
        i += 1


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
        print('[%s] => %d' % (value, i))
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i + 1] - row2[i + 1]) ** 2
    return sqrt(distance)


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# Make a prediction with KNN on skin Dataset
filename = 'glcm.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0]) - 1)
# define model parameter
num_neighbors = 5
# define a new record
row = [5.7, 2.9, 4.2, 1.3, 7, 5, 4, 3, 2, 2, 5, 3, 2]
# predict the label
label = predict_classification(dataset, row, num_neighbors)
print('Data=%s, Predicted: %s' % (row, label))
