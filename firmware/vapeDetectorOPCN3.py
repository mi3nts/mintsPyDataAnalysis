# ***************************************************************************
#  mintsXU4
#   ---------------------------------
#   Written by: Lakitha Omal Harindha Wijeratne
#   - for -
#   Mints: Multi-scale Integrated Sensing and Simulation
#   &
#   Algolook.com
#   ---------------------------------
#   Date: June 16th, 2019
#   ---------------------------------
#   This module is written for a classification of an Automated Vape Detector
#   --------------------------------------------------------------------------
#   https://github.com/mi3nts
#   http://utdmints.info/
#  ***************************************************************************

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from mintsDataAnalysis import *
import pickle
# Setting a random seed
np.random.seed(0)


def main():

    modelLocation = '../models/vapeClassifierOPCN3.sav'
    dataCSV = "../data/MINTS_001e06323952_OPCN3_2019_06_14.csv"

    print("-----------------------------------------------------")
    print("Multi-scale Integrated Sensing and Simulation (MINTS)")
    print('-----------------------------------------------------')
    print("Automated Vape Detector")
    print('-----------------------------------------------------')
    print(' ')


    #  Reading Training Data
    df = pd.read_csv(dataCSV)

    # Recognizing Feature Variables

    featureLabels = [\
                   'binCount0',\
                   'binCount1',\
                   'binCount2',\
                   'binCount3',\
                   'binCount4',\
                   'binCount5',\
                   'binCount6',\
                   'binCount7',\
                   'binCount8',\
                   'binCount9',\
                   'binCount10',\
                   'binCount11',\
                   'binCount12',\
                   'binCount13',\
                   'binCount14',\
                   'binCount15',\
                   'binCount16',\
                   'binCount17',\
                   'binCount18',\
                   'binCount19',\
                   'binCount20',\
                   'binCount21',\
                   'binCount22',\
                   'binCount23',\
                   'temperature',\
                   'humidity',\
                   'pm1',\
                   'pm2_5',\
                   'pm10',\
                   ]

    featuresDisplayed = [\
                   'Bin 0',\
                   'Bin 1',\
                   'Bin 2',\
                   'Bin 3',\
                   'Bin 4',\
                   'Bin 5',\
                   'Bin 6',\
                   'Bin 7',\
                   'Bin 8',\
                   'Bin 9',\
                   'Bin 10',\
                   'Bin 11',\
                   'Bin 12',\
                   'Bin 13',\
                   'Bin 14',\
                   'Bin 15',\
                   'Bin 16',\
                   'Bin 17',\
                   'Bin 18',\
                   'Bin 19',\
                   'Bin 20',\
                   'Bin 21',\
                   'Bin 22',\
                   'Bin 23',\
                   'Temperature',\
                   'Humidity',\
                   'PM$_1$',\
                   'PM$_{2.5}$',\
                   'PM$_{10}$',\
                   ]
    # Loading Features
    features=df[featureLabels]

    # Loading Classifier
    clf = pickle.load(open(modelLocation, 'rb'))

    # Gaining Predictions
    print('-----------------------------------------------------')
    print("-- Prediction Probabilty --")
    predictionProbabilty = clf.predict_proba(features)
    print(predictionProbabilty)

    print("-- Final Prediction --")
    prediction           = clf.predict(features)
    print(prediction)


if __name__ == "__main__":
   main()
