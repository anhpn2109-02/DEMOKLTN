import numpy as np
import pandas as pd
from sklearn.svm import SVC

data = pd.read_csv("heart_failure_clinical_records_dataset_clear.csv")
print(data)

X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

# chia nhỏ dữ liệu
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_test.shape, y_test.shape

# Mô hình 3: SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
svm_pred = svm_classifier.predict(X_test)
print("mô hình SVM:",svm_pred)

import pickle
pickle.dump(svm_classifier, open('model.pkl', 'wb'))