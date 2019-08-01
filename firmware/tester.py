import requests
import datetime
from mintsDataAnalysis import *

dateTime  = str(datetime.datetime.now())
occoredDate= dateTime.replace(" ","T")
print(str(occoredDate))


sensorDictionary = {
                      "cameraId": 7,
                      "ocurredDate": occoredDate,
                      "severityId": 1,
                      "situation": "Vape",
                      "statusId": 1
                    }





r = requests.post(\
                      url ="http://104.45.134.110:8080/api/alerts",\
                      json=sensorDictionary,\
                      auth=('algolook', 'safeai123')\
                      )
print("Status Code:" + str(r.status_code))

r1= requests.get(\
                'http://13.90.20.116:8080/api/v1/sensor/record/OPCN3/001e06323952',\
                auth=('algolook', 'safeai123')\
                )


r2 = requests.get('http://13.90.20.116:8080/api/v1/sensor/record/OPCN3/0242567a739f',auth=('algolook', 'safeai123'))
print("Status Code Get:" + str(r1.status_code))
print("Status Code Get:" + str(r1.status_code==200))


print("Status Code Get:" + str(r1.status_code))
print("Status Code Get:" + str(r1.status_code==200))

# print("Status Code Get:" + str(r1.text))
# print("Status Code Get:" + str(r1.json()))
