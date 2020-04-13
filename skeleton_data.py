#!/usr/bin/python

import os
import numpy as np



def dirToRAD(directory):
    """Converts a directory of pre-formatted skeleton data to single RelativeAnglesDistances format file"""
    #Open Directory
    trainFiles = [file for file in os.listdir(directory) if file.endswith(".txt")]
    for file in trainFiles:
        rawData = getRawData(directory+file)
        print('Raw',rawData)
    print("DIST")
    print(rawToDistAngle(rawData))



def rawToDistAngle(arr):
    """Takes an array of raw data and returns distance relative to joint 0 (in rawArr)
    In form: dist [head-hip, rhand-hip, rfoot-hip, lfoot-hip, lhand-hip],
    angle [head-rhand, rhand-rfoot, rfoot-lfoot, lfoot-lhand, lhand-head]"""

    shape = np.shape(arr)
    newArr = np.zeros((shape[0], 2, 5))
    for frame in range(shape[0]):
        vectors = []
        #Do Distance Calculations
        for joint in range(1,shape[1]):
            newArr[frame][0][joint-1] = np.linalg.norm(arr[frame][joint] - arr[frame][0])
            vectors.append(arr[frame][joint] - arr[frame][0]) #Gets the vectors of each joint-HipCenter
        #Do Angle Calculations
        for i in range(1,shape[1]):
            if i == shape[1]-1:
                cos_angle = np.dot(vectors[i-1], vectors[0]) / (np.linalg.norm(vectors[i-1]) * np.linalg.norm(vectors[0]))
            else:
                cos_angle = np.dot(vectors[i-1], vectors[i]) / (np.linalg.norm(vectors[i-1]) * np.linalg.norm(vectors[i]))
            newArr[frame][1][i-1] = np.arccos(cos_angle)
    print(newArr)



def getRawData(fileName):
    jointDict = {1:0,4:1,8:2,16:3,20:4,12:5}    #HipCenter, Head, RightHand, RightFoot, LeftFoot, LeftHand. Values just for indexing.
    f = open(fileName, 'r')
    frameNum = getFrameNumber(fileName, 20) #gets number of frames. Assuming 20 lines per joint
    rawData = np.zeros((frameNum, len(jointDict), 3))
    for line in f:
        words = line.split()
        if int(words[1]) in jointDict:          #Add new data
            frame = int(words[0])-1             #who starts indexes at 1 ew
            joint = jointDict[int(words[1])]
            x,y,z = words[2:]
            rawData[frame][joint] = float(x),float(y),float(z)
    f.close()
    return rawData


def getFrameNumber(fileName, jointNumber):
    """Gets number of frames in a file"""
    with open(fileName) as f:
        for i, l in enumerate(f):
            pass
        return (i+1)//jointNumber

if __name__ == '__main__':
    dirToRAD('dataset/train/')


