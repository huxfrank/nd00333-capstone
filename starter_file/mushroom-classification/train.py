from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory


def clean_data(data):
    
    #Label Encoding
    for col in data.columns:
    label_encoder = LabelEncoder()
    feature = data[str(col)]
    label_encoder.fit(feature)
    data[str(col)] = label_encoder.transform(feature)
    
    #Delete veil type because there is only one value for all entries
    del data['veil-type']
    
    #Split encoded data into independent and dependent variables to scale dependent data

    y_lc = df["class"]
    x_lc = df.drop("class", axis = 1)

    #Standardization of Features for Label Encoded Data
    scaler = StandardScaler()

    lc_x = pd.DataFrame(scaler.fit_transform(lc_x), columns = lc_x.columns)

    return x_lc,y_lc
    
#Create / Clean Dataset
datapath = "./mushroom_classification/mushrooms.csv"
ds = Dataset.Tabular.from_delimited_files(datapath)

x , y = clean_data(ds)
x_train, x_test, y_train, y_test = train_test_split(x,y)

#Run
run = Run.get_context()

def main():
    #Logging pertinent information and metrics
    run.log("Regularization Strength:", np.float(1.0))
    run.log("Max iterations:", np.int(100))

    model = LogisticRegression(C=1.0, max_iter=100).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(LogisticRegression, 'outputs/model.joblib')

if __name__ == '__main__':
    main()