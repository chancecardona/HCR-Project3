#!/usr/bin/python

import os, numpy as np
#import matplotlib.pyplot as plt            #Uncomment these if you have them and want more plotting features
#from mpl_toolkits.mplot3d import Axes3D


def dirToRAD(directory, toFile):
    """Converts a directory of pre-formatted skeleton data to single RelativeAnglesDistances format file"""
    #Open Directory
    trainFiles = [file for file in os.listdir(directory) if file.endswith(".txt")]
    fOut = open(toFile, 'w')
    for file in trainFiles:
        print(file)
        rawData = getRawData(directory+file)
        if 'rad' in toFile:
            data = rawToRAD(rawData)
        elif 'cust' in toFile:
            data = rawToCust(rawData)
        else:
            print('ERROR. Transformation to ' + toFile + ' failed. Please enter a string containing cust for custom representation or rad for RAD format.')
        #print('Raw:', rawData)
        #print('RAD:', RADdata)
        label = int(file[1:3])
        enumData = enumerate(getHistogram(data, file))
        fOut.write(str(label) + ' ')
        fOut.writelines([" %d:%s" % (index+1, elem) for index,elem in enumData])
        fOut.write('\n')
    fOut.close()

#TODO fix histograms. filter noise? get bins correct. get np.hist and plt.hist to do same thing.
def getHistogram(arr, fName, bins=10):
    dictJointName = {0:'Head to Hip',1:'RH-RF',2:'RF-LF',3:'LF-LH',4:'LH-Head'}
    fNum, typeNum, dNum = np.shape(arr)
    histArr = np.zeros((bins+bins)*dNum)
    for t in range(typeNum):
        #fig = plt.figure()
        for i in range(dNum):
            cur = arr[:,t,i]
            cur = cur[~np.isnan(cur)]
            #ax = fig.add_subplot(231+i)

            #1. use this for normal plotting
            #ax.plot(cur)

            #2. Use this for np histograms
            curHist, curBins = np.histogram(cur)
            #ax.bar(curBins[:-1] + np.diff(curBins)/2, curHist, np.diff(curBins))

            #3. Use this for matplotlib histograms
            #h = ax.hist(cur)
            #h = plt.hist(cur)
            #curHist = h[0]
            #curBins = h[1]

            #if t == 0:
            #    ax.set_title('dist: '+dictJointName[i])
            #else:
            #    ax.set_title('angle: '+dictJointName[i])
            #fig.suptitle(fName)
   #         print('CurHist'+str(i),curHist)
            histArr[(i+dNum*t)*bins:(i+1+dNum*t)*bins] = curHist / fNum #normalize and concatenate
        #plt.show()
   # print(histArr)
    return histArr



def rawToRAD(arr):
    """Takes an array of raw data and returns distance relative to joint 0 (in rawArr)
    In form: dist [head-hip, rhand-hip, rfoot-hip, lfoot-hip, lhand-hip],
    angle [head-rhand, rhand-rfoot, rfoot-lfoot, lfoot-lhand, lhand-head]"""

    shape = np.shape(arr) #frames, joints, xyz
    newArr = np.zeros((shape[0], 2, 5)) #frames, dist or angle, d's or theta's
    for frame in range(shape[0]):
        vectors = []
        #Do Distance Calculations
        for joint in range(1,6):
            newArr[frame][0][joint-1] = np.linalg.norm(arr[frame][joint] - arr[frame][0])
            vectors.append(arr[frame][joint] - arr[frame][0]) #Gets the vectors of each joint-HipCenter
        #Do Angle Calculations
        for i in range(1,6):
            if i == 5:
                cos_angle = np.dot(vectors[i-1], vectors[0]) / (np.linalg.norm(vectors[i-1]) * np.linalg.norm(vectors[0]))
            else:
                cos_angle = np.dot(vectors[i-1], vectors[i]) / (np.linalg.norm(vectors[i-1]) * np.linalg.norm(vectors[i]))
            newArr[frame][1][i-1] = np.arccos(cos_angle)
    return newArr

def rawToCust(arr):
    #dist:  hand to hand, hand to shoulder (l and r), lfoot to lhand (l and r)
    #angle: hand-elbow-shoulder (l and r), foot-knee-hip (l and r), elbow-shoulder-elbow
    jm = {'lhand':5, 'rhand':2, 'lelbow':7, 'relbow':8, 'lknee':9, 'rknee':10, 'lfoot':4, 'rfoot':3, 'shoulder':6, 'hip':0, 'head':1}
    shape = np.shape(arr)
    newArr = np.zeros((shape[0], 2, 5))
    for frame in range(shape[0]):
    #Distances
        newArr[frame][0][0] = np.linalg.norm(arr[frame][jm['lhand']] - arr[frame][jm['rhand']]) #Hand-Hand Dist
        newArr[frame][0][1] = np.linalg.norm(arr[frame][jm['lhand']] - arr[frame][jm['shoulder']]) #Left-Hand to ShoulderCenter
        newArr[frame][0][2] = np.linalg.norm(arr[frame][jm['rhand']] - arr[frame][jm['shoulder']]) #Right-Hand to ShoulderCenter
        newArr[frame][0][3] = np.linalg.norm(arr[frame][jm['lhand']] - arr[frame][jm['lfoot']]) #LeftHand to LeftFoot
        newArr[frame][0][4] = np.linalg.norm(arr[frame][jm['rhand']] - arr[frame][jm['rfoot']]) #RightHand to RightFoot
    #Angles
        lHtoE = arr[frame][jm['lhand']] - arr[frame][jm['lelbow']]
        rHtoE = arr[frame][jm['rhand']] - arr[frame][jm['relbow']]
        lEtoS = arr[frame][jm['lelbow']] - arr[frame][jm['shoulder']]
        rEtoS = arr[frame][jm['relbow']] - arr[frame][jm['shoulder']]
        newArr[frame][1][0] = np.arccos( np.dot(lHtoE, lEtoS) / (np.linalg.norm(lHtoE) * np.linalg.norm(lEtoS)) ) #Left Hand-Elbow-Shoulder
        newArr[frame][1][1] = np.arccos( np.dot(rHtoE, rEtoS) / (np.linalg.norm(rHtoE) * np.linalg.norm(rEtoS)) ) #Right ''
        lFtoK = arr[frame][jm['lfoot']] - arr[frame][jm['lknee']]
        rFtoK = arr[frame][jm['rfoot']] - arr[frame][jm['rknee']]
        lKtoH = arr[frame][jm['lknee']] - arr[frame][jm['hip']]
        rKtoH = arr[frame][jm['rknee']] - arr[frame][jm['hip']]
        newArr[frame][1][2] = np.arccos( np.dot(lFtoK, lKtoH) / (np.linalg.norm(lFtoK) * np.linalg.norm(lKtoH)) ) #Left Foot-Knee-Hip
        newArr[frame][1][3] = np.arccos( np.dot(rFtoK, rKtoH) / (np.linalg.norm(rFtoK) * np.linalg.norm(rKtoH)) ) #Right ''
        newArr[frame][1][4] = np.arccos( np.dot(rEtoS, lEtoS) / (np.linalg.norm(rEtoS) * np.linalg.norm(lEtoS)) ) #Elbow Shoulder Elbow
    return newArr



def getRawData(fileName):
    jointDict = {1:0,4:1,8:2,16:3,20:4,12:5, 3:6,6:7,10:8,14:9,18:10}    #HipCenter, Head, RightHand, RightFoot, LeftFoot, LeftHand. Values just for indexing.
    f = open(fileName, 'r')                                             #Shoulder center, LElbow, RElbow, lKnee, rKnee
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
    # Use these if you want to plot 3d data of the joints through all frames. Maybe make scatter to better see noise?
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot(rawData[:, 1, 0], rawData[:, 1, 1], rawData[:, 1, 2])
    return rawData


def getFrameNumber(fileName, jointNumber):
    """Gets number of frames in a file"""
    with open(fileName) as f:
        for i, l in enumerate(f):
            pass
        return (i+1)//jointNumber


#TODO filter data? Check if it removes NANs.




if __name__ == '__main__':
    dirToRAD('dataset/train/', 'rad_d1')
    #plt.show()
    print('Training RAD created.')
    dirToRAD('dataset/test/', 'rad_d1.t')
    print('Testing RAD created.')
    dirToRAD('dataset/train/', 'cust_d1')
    print('Training Cust created.')
    dirToRAD('dataset/test/', 'cust_d1.y')
    print('Testing Cust created.')



