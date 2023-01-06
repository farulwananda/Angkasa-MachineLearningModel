import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

glcm_df = pd.read_csv("glcmx5.csv")

print(glcm_df.shape)

# X -> features, y -> label


glcm_features = glcm_df[['dissimilarity_0', 'dissimilarity_45', 'dissimilarity_90', 'dissimilarity_135',
                         'correlation_0', 'correlation_45', 'correlation_90', 'correlation_135',
                         'homogeneity_0', 'homogeneity_45', 'homogeneity_90', 'homogeneity_135',
                         'contrast_0', 'contrast_45', 'contrast_90', 'contrast_135',
                         'ASM_0', 'ASM_45', 'ASM_90', 'ASM_135',
                         'energy_0', 'energy_45', 'energy_90', 'energy_135']].values

X = glcm_features

y = glcm_df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7)

print('X_train = ', X_train.shape)
print('X_test = ', X_test.shape)
print('y_train = ', y_train.shape)
print('y_test = ', y_test.shape)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X_train, y_train)

# predict the response for test
knn_pred_tr = knn.predict(X_train)
knn_pred_te = knn.predict(X_test)

# creating a confusion matrix
cm = confusion_matrix(y_test, knn_pred_te)
print(cm)

# Use score method to get accuracy of the model
print('----- Evaluation on Training Data -----')
score_tr = knn.score(X_train, y_train)
print('Accuracy Score: ', score_tr)
# Look at classification report to evaluate the model
print(classification_report(y_train, knn_pred_tr))
print('--------------------------------------------------------')
print('----- Evaluation on Test Data -----')
score_te = knn.score(X_test, y_test)
print('Accuracy Score: ', score_te)
# Look at classification report to evaluate the model
print(classification_report(y_test, knn_pred_te))
print('--------------------------------------------------------')
