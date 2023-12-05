{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "158a48d2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "GaussianNB model :\n",
      " accuracy: 1.0\n",
      "recall_score: 1.0\n",
      "f1_score: 1.0\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<mlflow.models.model.ModelInfo at 0x15610572c90>"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import os\n",
    "import sys\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn import preprocessing\n",
    "from sklearn.metrics import accuracy_score,recall_score,f1_score\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "import warnings\n",
    "\n",
    "\n",
    "import mlflow\n",
    "import mlflow.sklearn\n",
    "\n",
    "import logging\n",
    "logging.basicConfig(level=logging.WARN)\n",
    "logger = logging.getLogger(__name__)\n",
    "\n",
    "def eval_metrics(actual, pred):\n",
    "    accuracy = accuracy_score(test_y, y_pred)\n",
    "    recall_score = recall_score(test_y, y_pred)\n",
    "    f1_score = f1_score(test_y, y_pred)\n",
    "    return accuracy,recall_score,f1_score\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    warnings.filterwarnings(\"ignore\")\n",
    "    np.random.seed(40)\n",
    "\n",
    "# Read pointure csv file\n",
    "data = pd.read_csv('pointure.data')\n",
    "\n",
    "# PRE-TRAITEMENT DES DONNÉES\n",
    "label_encoder = preprocessing.LabelEncoder()\n",
    "input_classes = ['masculin','féminin']\n",
    "label_encoder.fit(input_classes)\n",
    "\n",
    "# transformer un ensemble de classes\n",
    "encoded_labels = label_encoder.transform(data['Genre'])\n",
    "data['Genre'] = encoded_labels\n",
    "\n",
    "\n",
    "\n",
    "# Split the data into training and test sets. (0.75, 0.25) split.\n",
    "train, test = train_test_split(data)\n",
    "\n",
    "# The predicted column is \"Genre\" which is a scalar from [3, 9]\n",
    "train_x = train.drop([\"Genre\"], axis=1)\n",
    "test_x = test.drop([\"Genre\"], axis=1)\n",
    "train_y = train[[\"Genre\"]]\n",
    "test_y = test[[\"Genre\"]]\n",
    "\n",
    "mlflow.set_tracking_uri(\"http://localhost:5000\")\n",
    "mlflow.set_experiment(experiment_name='experiment1_examen')\n",
    "\n",
    "with mlflow.start_run():\n",
    "    gnb = GaussianNB()\n",
    "    gnb.fit(train_x, train_y)\n",
    "\n",
    "y_pred = gnb.predict(test_x)\n",
    "(test_y, y_pred)\n",
    "accuracy = accuracy_score(test_y,y_pred)\n",
    "recall_score = recall_score(test_y,y_pred)\n",
    "f1_score = f1_score(test_y,y_pred)\n",
    "\n",
    "print(\"GaussianNB model :\")\n",
    "print(\" accuracy: %s\"  % accuracy )\n",
    "print(\"recall_score: %s\" % recall_score)\n",
    "print(\"f1_score: %s\" % f1_score)\n",
    "\n",
    "# MLFLOW\n",
    "mlflow.log_metric(\"accuracy\", accuracy)\n",
    "mlflow.log_metric(\"recall_score\", recall_score)\n",
    "mlflow.log_metric(\"f1_score\", f1_score)\n",
    "\n",
    "mlflow.sklearn.log_model(gnb, \"model\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b76b935d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
