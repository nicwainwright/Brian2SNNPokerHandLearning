# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 03:24:55 2018

@author: Nic

data manipulation and loading
"""

import numpy as np
import time

def oneHot(label,possibleLabels):
    oneHot = np.zeros(possibleLabels)
    oneHot[label] = 1
    return list(oneHot) #gets one-hot list representation of label

def loadTraining():
    start = time.time()
    trainFile = 'poker-hand-training-true.data' #this file must be in same directory as this loadFunctions.py
    with open(trainFile) as f:
        trainData = f.read().splitlines() #gets the hands in a list of strings where each element is of format ['0,1,...2','2,3,...3',...]
    splitString = [i.split(',') for i in trainData] #splits each element of list into a list of strings of format [['0','1',...'2'],['2','3',...'3'],...]
    hands = np.asarray([[int(i) for i in x] for x in splitString]) #converts every single string into an int and stores in numpy array size (25010,11)
    
    cardHotVersion = getCardsAndHot(hands) #converts the (25010,11) list of lists into a list of tuples where the first item is card number 0-51 and second item is oneHot of label
    end = time.time()
    print 'time needed to load training set:', end - start
    return cardHotVersion

def loadTesting():
    start = time.time()
    testFile = 'poker-hand-testing.data' #this file must be in same directory as this loadFunctions.py
    with open(testFile) as f:
        trainData = f.read().splitlines() #gets the hands in a list of strings where each element is of format ['0,1,...2','2,3,...3',...]
    splitString = [i.split(',') for i in trainData] #splits each element of list into a list of strings of format [['0','1',...'2'],['2','3',...'3'],...]
    hands = np.asarray([[int(i) for i in x] for x in splitString]) #converts every single string into an int and stores in numpy array size (25010,11)
    cardHotVersion = getCardsAndHot(hands) #converts the (25010,11) list of lists into a list of tuples where the first item is card number 0-51 and second item is oneHot of label
    end = time.time()
    print 'time needed to load testing set:', end - start
    return cardHotVersion

def getCardsAndHot(dataList):
    cardsAndHots = []
    for i in range(len(dataList)):
        cardHot = []
        for x in range(0,10,2):
            card = ((dataList[i][x]-1)*13) + (dataList[i][x+1]-1) # (suit-1)*13 + (card-1)
            cardHot.append(card) #add cards 0-4 to the list of cards
        cardsAndHots.append((cardHot,oneHot(dataList[i][10],10))) # add card and onehot tuple to list to return
    return cardsAndHots #a list of tuples. each tuple is the 5 cards (0-51) and the one hot nparray representation of label