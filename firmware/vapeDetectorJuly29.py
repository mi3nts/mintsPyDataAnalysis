# ***************************************************************************
#  mintsXU4
#   ---------------------------------
#   Written by: Lakitha Omal Harindha Wijeratne
#   - for -
#   Mints: Multi-scale Integrated Sensing and Simulation
#   &
#   Algolook.com
#   ---------------------------------
#   Date: June 20th, 2019
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
import time
import datetime
import requests


def readFeaturesPatch(argsOut):
    import json
    moveForward = True
    if(argsOut["local"]):
        print("--Reading Local Data--")
        with open(argsOut["dataSource"], 'r') as myfile:
            dfPre=json.load(myfile)
    else:
        print("--Reading Data From External Servers--")
        dfPre, moveForward =  getJsonURL(argsOut)

    if(moveForward):
        df = pd.DataFrame.from_dict(dfPre, orient='index')
        # print(df.binCount0)
        return df,moveForward;
    else:
        print("No Recent Data Found")
        return "xxxx",moveForward;



def main():

    printIntro("Automated Vape Detector")
    argsOut =  readArgs()
    clf = pickle.load(open(argsOut["modelPath"], 'rb'))
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

    print("------------------------------")
    while(True):
        time.sleep(argsOut["frequency"])
        try:
            print("------TEST----------")
            df, moveForward = readFeaturesPatch(argsOut)
            print(df[0]['binCount0'])
            df[0]['binCount0']=df[0]['binCount0']/10
            df[0]['binCount1']=df[0]['binCount1']/10
            df[0]['binCount2']=df[0]['binCount2']/10
            print(df)
            if(moveForward):
                prediction = getPredictionOPCN3(df,clf,featureLabels)
                if(prediction==1):
                    sendAlert(df,argsOut)

                printMints()

        except Exception as e:
            print(str(e))




if __name__ == "__main__":
   main()
