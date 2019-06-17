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
import json


def main():

    modelLocation = '../models/vapeClassifierOPCN3.sav'
    dataCSV       = "../data/MINTS_001e06323952_OPCN3_2019_06_14.csv"
    jsonFile      = "../data/OPCN3.json"
    print(" ")
    print("-----------------------------------------------------")
    print("Multi-scale Integrated Sensing and Simulation (MINTS)")
    print('-----------------------------------------------------')
    print("Automated Vape Detector")
    print('-----------------------------------------------------')
    print(' ')

    try:
        with open(jsonFile, 'r') as myfile:
            dfPre=json.load(myfile)


        # Recognizing Feature Variables
        # print(dfPre)
        df = pd.DataFrame.from_dict(dfPre, orient='index')

        print("Features Read:")
        print(df)

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

        # Loading Features
        features=[df[0][featureLabels]]

        # # Loading Classifier
        clf = pickle.load(open(modelLocation, 'rb'))

        # Gaining Prediction
        predictionProbabilty = clf.predict_proba(features)
        print("  ")
        print('-----------------------------------------------------')
        print("-- Prediction Probabilty --")
        print("Clean Air Probabilty: " + str(getColumn(predictionProbabilty,0)[0]))
        print("Juul Vape Probabilty: " + str(getColumn(predictionProbabilty,1)[0]))
        print("Lysol     Probabilty: " + str(getColumn(predictionProbabilty,2)[0]))
        print("Febreze   Probabilty: " + str(getColumn(predictionProbabilty,3)[0]))
        print("Breath    Probabilty: " + str(getColumn(predictionProbabilty,4)[0]))
        print("  ")
        print('-----------------------------------------------------')
        print("-- Final Prediction --")
        prediction           = clf.predict(features)
        targetDisplayLabels = ['Clean','Juul','Lysol','Febreze','Breath']

        print("Final Prediction: " + str(targetDisplayLabels[int(prediction)]))

        print("  ")
        print("-----------------------------------------------------")
        print("Multi-scale Integrated Sensing and Simulation (MINTS)")
        print('-----------------------------------------------------')

    except:
        print("Corrupt JSON File")
if __name__ == "__main__":
   main()
