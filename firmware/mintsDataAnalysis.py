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



def plotFeatureImportainces(features,importances,title):
    import matplotlib.pyplot as plt
    import numpy as np
    indices = np.argsort(importances)
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools


    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def getColumn(matrix, i):
    return [row[i] for row in matrix]




def printIntro(intro):

    printMints()
    print(intro)
    print('-----------------------------------------------------')
    print(' ')


def printMints():

    print(" ")
    print("-----------------------------------------------------")
    print("Multi-scale Integrated Sensing and Simulation (MINTS)")
    print('-----------------------------------------------------')




def readArgs():
    import argparse
    import os
    from distutils.util import strtobool
    import numbers


    #set up command line arguments
    parser = argparse.ArgumentParser(description="-- Vape Classification --")
    parser.add_argument('-m','--model', dest='modelPath', help="Path to OPC Model. (e.g. '-m ../models/vapeClassifierOPCN3.sav')")
    parser.add_argument('-d','--destination', dest='destination', help="Destination URL to publish Vape events. (e.g. '-h www.google.com')")
    parser.add_argument('-u','--user', dest='userName', help="Username for API Authentication (e.g. '-u adam')")
    parser.add_argument('-p','--password', dest='password', help="Password for API Authentication (e.g. '-p adamsPW')")
    parser.add_argument('-f','--frequency', dest='frequency',help="Frequency to check incoming data in Seconds (e.g. '-f 1')")
    parser.add_argument('-l','--local', dest='local',help="Specify wheather the raw data is Local or External (e.g. '-l True' or '-l False')")
    parser.add_argument('-s','--source', dest='dataSource',help="Data Source Location (e.g. '-s ../data/OPCN3.json' or '-s http://13.90.20.116:8080/api/v1/sensor/record')")
    parser.add_argument('-n','--node', dest='nodeID',help="Specify the Node ID (e.g. '-n 0242567a739f')")
    parser.add_argument('-e','--sensor', dest='sensorID',help="Specify the Sensor ID (e.g. '-e OPCN3')")
    args = parser.parse_args()

    # make sure user specifies a Model Path
    if args.modelPath== None:
        print("Error: No Model Path given.")
        exit(1)

    # make sure user specifies a destination URL
    if args.destination== None:
        print("Error: No Destination URL Given")
        exit(1)

    # make sure user specifies a userName
    if args.userName== None:
        print("Error: No User Name Given for API Authentication")
        exit(1)

    # make sure user specifies a PW
    if args.password== None:
        print("Error: No Password Given for API Authentication")
        exit(1)

    # make sure user specifies wheather the raw data local or external
    if args.local== None:
        print("Error: No specification on local or external data preference")
        exit(1)

    # make sure user specifies wheather the raw data local or external
    if args.dataSource== None:
        print("Error: No data source specified")
        exit(1)

    if args.nodeID== None:
        print("Error: No Node ID specified")
        exit(1)

    if args.sensorID== None:
        print("Error: No Sensor ID specified")
        exit(1)

    modelPath   = str(args.modelPath)
    dataSource  = str(args.dataSource)

    # Check if the Model Exists
    if os.path.isfile(modelPath):
        if not modelPath.endswith(".sav"):
            print("Error: '" + str(modelPath) + "' not a model file!")
            exit(1);
    else:
        print("Error: '" + str(modelPath) + "' does not exist!")
        exit(1);

    localPre  = strtobool(str(args.local))


    if localPre==1:
        local = True
        if os.path.isfile(args.dataSource):
            if not dataSource.endswith(".json"):
                print("Error: '" + str(dataSource) + "' not a Json Object!")
                exit(1);
        else:
            print("Error: '" + str(dataSource) + "' does not exist locally!")
            exit(1);

    elif localPre==0:
        local = False
        if(not(dataSource.startswith("http://"))):
            print("Error: Invalid Data Source URL("+dataSource+")")
            exit()



    else:
        print("invalid Specification on Local or External Data Preference")
        exit(1)

    frequencyPre  = args.frequency

    # set Frequency
    if frequencyPre== None:
        frequency = 1
        print("Setting Default Frequency to 1 Second")

    elif(not(frequencyPre.isdigit())):
        print("Given frequency("+ str(args.frequency) + ") is not a Positive Integer" )
        exit(1)


    else:
        frequency = int(args.frequency)
        # print("Setting Frequency to "+ str(frequency) + " Second(s)" )
        # exit(1)


    destination = args.destination
    userName    = args.userName
    password    = args.password
    nodeID      = args.nodeID
    sensorID    = args.sensorID


    print("Model Path     : " + modelPath )
    print("Raw Data Source: " + dataSource)
    print("Destination    : " + destination )
    print("User Name      : " + userName )
    print("Password       : " + password )
    print("Node ID        : " + nodeID )
    print("Sensor ID      : " + sensorID )
    print("Frequency      : " + str(frequency) + " seconds(s)" )
    print("Local Raw Data : " + str(local) )

    argsOut = {\
                "modelPath": modelPath,\
                "dataSource":dataSource,\
                "destination":destination,\
                "userName":userName,\
                "password":password,\
                "nodeID":nodeID,\
                "sensorID":sensorID,\
                "frequency":frequency,\
                "local":local }

    return argsOut;

def getJsonURL(argsOut):
    import requests

    urlGet =   argsOut["dataSource"] +"/"+ argsOut["sensorID"]+"/"+ argsOut["nodeID"]

    jsonData = requests.get(\
                            url = urlGet,\
                            auth=(argsOut["userName"],\
                                  argsOut["password"]\
                                  )\
                           )
    if(jsonData.status_code==200):
        return jsonData.json(),True;
    else:
        return "xxxxx",False;



def sendAlert(df,argsOut):
    print("--Vape Detected--")

    dateTime  = str(df[0]['dateTime'])
    ocurredDate= dateTime.replace(" ","T")

    sensorDictionary = {
                          "cameraId": 7,
                          "ocurredDate": ocurredDate,
                          "severityId": 1,
                          "situation": "Vape",
                          "statusId": 1
                        }


    urlGet =   argsOut["destination"]

    r = requests.post(\
                      url =argsOut["destination"],\
                      json=sensorDictionary,\
                      auth=(argsOut["userName"],\
                            argsOut["password"])\
                      )
    if(r.status_code==200):
        print("Alert Posted")
    else:
        print("Alert Not posted: Status Code "+ str(r.status_code) )



def readFeatures(argsOut):
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
        return df,moveForward;
    else:
        print("No Recent Data Found")
        return "xxxx",moveForward;


def getPredictionOPCN3(df,clf,featureLabels):
    features=[df[0][featureLabels]]

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
    print('-----------------------------------------------------')
    return int(prediction)
