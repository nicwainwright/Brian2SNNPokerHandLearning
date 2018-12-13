# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 00:17:42 2018
main.py
@author: Nic
"""
import brianFunctions as bf
import loadFunctions as lf
import numpy as np
import random

training = lf.loadTraining() #list of tuples where the first item is card number 0-51 and second item is oneHot of label
#testing  = lf.loadTesting() # gets the massive testing set

hands = [item[0] for item in training]#get a list of lists, where each item is the cards in the hand from training
labels = [item[1] for item in training] #get a list of lists, where each item is the oneHot label of the hand from training

# get the first n examples to put through training
handsFirst = hands[:1]
labelsFirst = labels[:1]

#create layers. input is length 52 (number of cards in deck), middle layer is arbitrary, output is 10 (possible hand types from high card to royal flush)
input_layer, hidden_layer, output_layer = bf.createLayers(5,1,5)
#get list of synapses between the layers i&j and j&k
S_ij, S_jk = bf.createSynapses(input_layer, hidden_layer, output_layer)
#first_hand=training[0][0]
#run training on these objects, passing in the hands and labels we want to use to train
#saves the final weights as a list of 1-value arrays to ij and jk, to be save to file for initializing a training array

errorList = []
errorSumList = []
handsFirst = ([0,3,2,1,4],[4,3,2,0,1],[3,4,1,2,0],[2,0,1,4,3],[1,1,2,3,0])
labelsFirst = ([0,1,0,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,1,0,0,0],[1,0,0,0,0])
ij,jk = bf.runMonitor(handsFirst,labelsFirst,input_layer, hidden_layer, output_layer, S_ij, S_jk,errorList,errorSumList)


#training[0][1]  -->> replace with label
