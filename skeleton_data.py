#!/usr/bin/python

import os

#Open Directory
trainFiles = [file for file in os.listdir("dataset/train") if file.endswith(".txt")]
for file in trainFiles:
    f = open('dataset/train/'+file, 'r')
    for line in f:
        print(line[0], line[1])
        if line[0] == frame:
            if line[1] == 1:    #HipCenter

            elif line[1] == 4:  #Head

            elif line[1] == 8:  #Right Hand

            elif line[1] == 12: #Left Hand

            elif line[1] == 16: #Right Foot

            elif line[1] == 20: #Left Foot

    f.close()
