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

    dataCSV = "../data/OPC_Vape.csv"
    predictedCSV = dataCSV.replace(".csv","_Prediction.csv")
    filename = '../models/vapeClassifierOPCN3.sav'

    print(" ")
    print("-----------------------------------------------------")
    print("Multi-scale Integrated Sensing and Simulation (MINTS)")
    print('-----------------------------------------------------')
    print("Automated Vape Detector")
    print('-----------------------------------------------------')
    print(' ')
    print('-----------------------------------------------------')
    print("Reading CSV data from " +  str(dataCSV))

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

    print(" ")
    print('-----------------------------------------------------')
    print("Dividing into Training and Test Data")

    allFeatures=df[featureLabels]

    #  Reading Target Variable
    allTargetLabels=  df.iloc[:,-1]
    allTargets = pd.factorize(allTargetLabels)[0]

    # Dividing into training and testing
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
    trainDf, testDf = df[df['is_train']==True], df[df['is_train']==False]

    trainFeatures      = trainDf[featureLabels]
    testFeatures       = testDf[featureLabels]
    trainTargetsLabels = allTargetLabels[df['is_train']==True]
    testTargetLabels   = allTargetLabels[df['is_train']==False]
    trainTargets       = allTargets[df['is_train']==True]
    testTargets        = allTargets[df['is_train']==False]

    print(" ")
    print('-----------------------------------------------------')
    print("Training a Random Forest Classifier")

    clf = RandomForestClassifier(n_jobs=2, random_state=0)

    # Training the Classifier
    clf.fit(trainFeatures, trainTargets)

    print(" ")
    print('-----------------------------------------------------')
    print("Saving the trainined Random Forest Classifier @ :" + filename)
    pickle.dump(clf, open(filename, 'wb'))

    print('-----------------------------------------------------')
    print("Gaining Predictions........................")
    print(" ")
    trainPrediction =  clf.predict(trainFeatures)
    testPrediction  =  clf.predict(testFeatures)
    allPrediction   =  clf.predict(allFeatures)

    trainPreditedNames = allTargets[trainPrediction]
    testPreditedNames  = allTargets[testPrediction]
    allPreditedNames   = allTargets[allPrediction]

    trainPredictionProbabilty = clf.predict_proba(trainFeatures)
    testPredictionProbabilty  = clf.predict_proba(testFeatures)
    allPredictionProbabilty   = clf.predict_proba(allFeatures)


    print('-----------------------------------------------------')
    print("Printing Predictor importances........................")
    print(" ")



    for name, importance in zip(featureLabels, clf.feature_importances_):
        print(name, "=", importance)

    plotFeatureImportainces(featuresDisplayed,\
                                clf.feature_importances_,\
                                    "Feature Importances - OPCN3")
    print(" ")
    
    print('-----------------------------------------------------')
    print("Gaining Confusion Matrices...........................")
    print(" ")
    trainConfusion= confusion_matrix(trainTargets,trainPrediction)
    testConfusion= confusion_matrix(testTargets,testPrediction)
    allConfusion= confusion_matrix(allTargets,allPrediction)

    print('-----------------------------------------------------')
    print("Confusion Matrix for Random Forest Classifier - Training")
    print(trainConfusion)
    print(" ")
    plot_confusion_matrix(trainConfusion,
                          normalize    = False,
                          target_names = ['Clean', 'Juul','Lysol','Febreze','Breath'],
                          title        = "Training Data - OPCN3")

    print('-----------------------------------------------------')
    print("Confusion Matrix for Random Forest Classifier - Testing")
    print(testConfusion)
    print(" ")
    plot_confusion_matrix(testConfusion,
                          normalize    = False,
                          target_names = ['Clean', 'Juul','Febreze','Breath'],
                          title        = "Testing Data - OPCN3")

    print('-----------------------------------------------------')
    print("Confusion Matrix for Random Forest Classifier - All Data")
    print(allConfusion)
    print(" ")

    plot_confusion_matrix(allConfusion,
                          normalize    = False,
                          target_names = ['Clean', 'Juul','Lysol','Febreze','Breath'],
                          title        = "All Data - OPCN3")


    df['prediction']    = allPrediction
    targetDisplayLabels = {0:'Clean', 1:'Juul',2:'Lysol',3:'Febreze',4:'Breath'}

    df['predictionLabels']  = df['prediction'].map(targetDisplayLabels)

    df['cleanProb'] = getColumn(allPredictionProbabilty,0)
    df['juulProb'] = getColumn(allPredictionProbabilty,1)
    df['lysolProb'] = getColumn(allPredictionProbabilty,2)
    df['febrezeProb'] = getColumn(allPredictionProbabilty,3)
    df['breathProb'] = getColumn(allPredictionProbabilty,4)

    print('-----------------------------------------------------')
    print("Saving Predictions @ :" + predictedCSV)
    print(" ")
    df.to_csv (predictedCSV, index = None, header=True)

    print("-----------------------------------------------------")
    print("Multi-scale Integrated Sensing and Simulation (MINTS)")
    print('-----------------------------------------------------')


if __name__ == "__main__":
   main()
