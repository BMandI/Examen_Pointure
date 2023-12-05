import os
import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import warnings


import mlflow
import mlflow.sklearn

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
accuracy = accuracy_score(test_y, y_pred)
recall_score = recall_score(test_y, y_pred)
f1_score = f1_score(test_y, y_pred)
return accuracy,recall_score,f1_score

if __name__ == "__main__":
warnings.filterwarnings("ignore")
np.random.seed(40)

# Read pointure csv file
data = pd.read_csv('pointure.data')

# PRE-TRAITEMENT DES DONNÉES
label_encoder = preprocessing.LabelEncoder()
input_classes = ['masculin','féminin']
label_encoder.fit(input_classes)

# transformer un ensemble de classes
encoded_labels = label_encoder.transform(data['Genre'])
data['Genre'] = encoded_labels

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)

# The predicted column is "Genre" which is a scalar from [3, 9].
train_x = train.drop(["Genre"], axis=1)
test_x = test.drop(["Genre"], axis=1)
train_y = train[["Genre"]]
test_y = test[["Genre"]]

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_name='experiment_examen')

with mlflow.start_run():
gnb = GaussianNB()
gnb.fit(train_x, train_y)

y_pred = gnb.predict(test_x)
(test_y, y_pred)
accuracy = accuracy_score(test_y,y_pred)
recall_score = recall_score(test_y,y_pred)
f1_score = f1_score(test_y,y_pred)
print("GaussianNB model :")
print(" accuracy: %s" % accuracy)
print("recall_score: %s" % recall_score)
print("f1_score: %s" % f1_score)

mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("recall_score", recall_score)
mlflow.log_metric("f1_score", f1_score)

mlflow.sklearn.log_model(gnb, "model")
