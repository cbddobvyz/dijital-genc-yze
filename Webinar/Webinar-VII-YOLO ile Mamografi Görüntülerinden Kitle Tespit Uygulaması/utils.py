import os
import os
import glob
import cv2
import utils
import pydicom as dicom
import plistlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def create_data_dir(folderPath):
    if not os.path.exists(folderPath):
        os.mkdir(folderPath)

def draw_annotation(image, line):
    centerPointX = float(line[1]) * image.shape[1]
    centerPointY = float(line[2]) * image.shape[0]
    
    width = float(line[3]) * image.shape[1]
    height = float(line[4]) * image.shape[0]
    
    startPoint = (int(centerPointX-(width/2)), int(centerPointY-(height/2)))
    endPoint = (int(centerPointX+(width/2)), int(centerPointY+(height/2)))

    color = (255,0,0)
    thickness = 5
    
    image = cv2.rectangle(image, startPoint, endPoint, color, thickness)


def saveImages(dicomFilePath, saveFolderPath):
    """
    Saving dicom images as png

    Parameters
    ---------- 
    dicomFilePath : str
        Dicom File Path
    saveFolderPath : str
        Image Save Folder Path
    
    Returns
    -------
    
    imageFilePath: str
        Saved Image File Path
        
    """
    
    dicomData = dicom.read_file(dicomFilePath)
    imgArray = dicomData.pixel_array
    imgArray = 255*((imgArray - imgArray.min()) / (imgArray.max() - imgArray.min()))
    imgArray = imgArray.astype(np.uint8)
    imgArray = Image.fromarray(imgArray).convert("L")
    
    dicomFileName = dicomFilePath.split(os.path.sep)[-1]
    patientID = '_'.join(dicomFileName.split("_")[:2])
    view = ''.join(dicomFileName.split("_")[3:5])
    imageFileName = '_'.join([patientID, view]) + '.png'
    imageFilePath = os.path.join(saveFolderPath, imageFileName)
    imgArray.save(imageFilePath)
    
    return imageFilePath


def loadPoint(point_string):
    x, y = tuple([float(num) for num in point_string.strip('()').split(',')])
    return y, x


def saveMaskYOLOFormat(xmlFilePath, imageFilePath, saveMaskPath, biradsScore, imshape):
    
    """
    Saving mask informations to text file

    Parameters
    ---------- 
    xmlFilePath : str
        XML File Path
    imageFilePath : str
        Image File Path
    saveMaskPath : str
        Save Mask File Path
    biradsScore : int
        Malignancy-Benignance or Mass
    imshape : tupple
        Image Shape
    Returns
    -------
    
    """
    
    with open(xmlFilePath, 'rb') as maskFile:
        plistDict = plistlib.load(maskFile, fmt=plistlib.FMT_XML)['Images'][0]
        
    pngFileName = imageFilePath.split(os.path.sep)[-1] 
    
    rois = plistDict['ROIs']
    maskPoints = []
    for roi in rois:
        numPoints = roi['NumberOfPoints']
        if numPoints == 1:
            continue
        
        abnormality = roi["Name"]
                        
        if abnormality =="Mass":
            
            points = roi['Point_px']
            points = [loadPoint(point) for point in points]
            points = np.array(points)
            
            xmin = int(min(points[:,1]))
            ymin = int(min(points[:,0]))
            xmax = int(max(points[:,1]))
            ymax = int(max(points[:,0]))

            normalized_center_x = (xmin + (xmax-xmin)/2) / imshape[0]
            normalized_center_y = (ymin + (ymax-ymin)/2) / imshape[1]
            normalized_width = (xmax-xmin) / imshape[0]
            normalized_height = (ymax-ymin) / imshape[1]
            
            maskPoints.append(' '.join([str(biradsScore), str(normalized_center_x), str(normalized_center_y), str(normalized_width), str(normalized_height)]))

        with open(saveMaskPath, 'w') as textFile:
            textFile.write('\n'.join(maskPoints))
            textFile.write('\n')