import pandas as pd
import numpy as np
from os import listdir
from unidecode import unidecode


cats = ['Rooms', 'Date', 'Location', 'Service', 'Business service', 'Author', 'Check in / front desk', 'No. Helpful', 'Cleanliness', 'Content', 'Value', 'No. Reader', 'Overall']


def getBlankFrame():
    
    data = pd.DataFrame(columns=cats)
    
    return data


def addFileToData(filename, data):
    intColumns = ['No. Reader', 'No. Helpful', 'Cleanliness','Check in / front desk', 'Value', 'Overall', 'Service', 'Business service', 'Rooms', 'Location']
    characterThreshold = 60
    with open(filename, 'r') as content_file:
        content = content_file.read()
        
        #print(repr(content))
    if content.count("\r") > 0:
        reviews = content.split("\r\n\r\n")
    else:
        reviews = content.split("\n\n")
    
    for r in reviews:
        thisReview = pd.Series([None]*len(cats), cats)
        splt = r.split("\n")
        for s in splt:
            for c in cats:
                if "<"+c+">" in s:
                    value = s.replace('<'+c+'>', '')
                    if c in intColumns:
                        value = int(value)
                    if value == -1: #we dont want -1 as this is going to mess up averaging, take np.nan
                        value = np.nan

                    if c == "Content":
                        value = remove_non_ascii(value.lower())

                    thisReview[c] = value
                    
        if not thisReview["Content"] == None and len(thisReview["Content"]) > characterThreshold:
            #only add if theres content and its long enough
            data = data.append(thisReview, ignore_index=True)
    return data

def getStandardData(numFiles = 10):
    files = sorted(listdir('Review_Texts/'))
    df = getBlankFrame()

    for file in files[:numFiles]:
        df = addFileToData('Review_Texts/'+file, df)

    return df
      

def remove_non_ascii(text):
    return unidecode(unicode(text, encoding="utf-8"))