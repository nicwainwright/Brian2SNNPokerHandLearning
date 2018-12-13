# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 23:59:51 2018

@author: Nic

Credit to Dr. Craig Henriquez from Duke University for the skeleton code for 
a 3 izhikevich-neuron NeuronGroup on which these neurons were based

Brian2 Functions for use in main
"""
from random import random
from brian2 import * #analysis:ignore
from matplotlib import pyplot as plt
import numpy as np
import loadFunctions as lf
set_device('runtime')
prefs.codegen.target = 'cython'
defaultclock.dt=.01*ms

alpha = 0.04 #learning rate 

# Generic parameters that we found to work and cause spikes as seen in Duke BME503 Homework 3 - Part1
area = 20000*umetre**2
Cm = 1*ufarad*cm**-2
El = -65*mV
Esyn=0*mV
gl = 0.7
tau_ampa=0.8*ms
g_synpk=0.08
w_works = g_synpk/(tau_ampa*exp(-1))*ms # this value of weight works to create spikes 
#Izhikevich Parameters 
a =  .02
b = .25
c = -65*mV
d = 2*mV

#leakage current
eqs_il = '''
ileak = gl * (El-v) : volt
'''

#standard Izhikevich model of a neuron with g_ampa scaled to be functional
eqs = '''
dv/dt = 0.04*v**2/mV/ms + 5*v/ms + 140*mV/ms - u/ms+ I/area/Cm + 3000*g_ampa*(Esyn-v)/ms:  volt
du/dt = a*(b*v - u)/ms : volt

dg_ampa/dt = -g_ampa/tau_ampa + z: 1 # this is a exponential synapse
dz/dt = -z/tau_ampa : Hz

I : amp
''' 
eqs+=eqs_il


def sigmoid(x,prime=False): #standard sigmoid function
    if(prime==True):
        return (x*(1-x))
    return 1/(1*np.exp(-x))

def createLayers(num_inputs, num_hiddens, num_outputs):
    #create a neuron group for the input, hidden, and output layers
    # Threshold and refractoriness are only used for spike counting
    input_layer = NeuronGroup(num_inputs, eqs,threshold='v > 10*mV',reset='v = c;u=u+d', method='euler')
    input_layer.v = -80*mV
    hidden_layer = NeuronGroup(num_hiddens, eqs,threshold='v > 10*mV',reset='v = c;u=u+d', method='euler')
    hidden_layer.v = -80*mV
    output_layer = NeuronGroup(num_outputs, eqs,threshold='v > 10*mV',reset='v = c;u=u+d', method='euler')
    output_layer.v = -80*mV
    return input_layer, hidden_layer, output_layer

def createSynapses(input_layer,hidden_layer,output_layer):
    num_inputs = len(input_layer)
    num_hidden = len(hidden_layer)
    num_output = len(output_layer)
    #create a list of synapses connecting each input neuron to each hidden neuron
    S_ij=[]
    for I in range(num_inputs):
        for J in range(num_hidden):
            print 'Creating Synapse from Input',I,'to hidden',J
            newSyn = Synapses(input_layer, hidden_layer,model='''w_ij:1 ''',on_pre='''z += w_ij*Hz''')
            newSyn.connect(i = [I],j=[J])
            newSyn.w_ij=w_works
            S_ij.append(newSyn)
    #create a list of synapses connecting each hidden neuron to each output neuron        
    #order of Synapses for this group is j0k0,j0k1,...j1k0,j1k1,...jnkn
    S_jk=[]
    for J in range(num_hidden):
        for K in range(num_output):
            print 'Creating Synapse from Hidden',J,'to output',K
            newSyn = Synapses(hidden_layer, output_layer,model='''w_jk:1 ''',on_pre='''z += w_jk*Hz''')
            newSyn.connect(i = [J],j=[K])
            newSyn.w_jk=w_works
            S_jk.append(newSyn)
    
    return S_ij, S_jk #return the lists of Synapse objects

def runMonitor(cards,label, input_layer, hidden_layer, output_layer, S_ij, S_jk,errorList,errorSumList):

    mon=StateMonitor(output_layer,('v', 'g_ampa'),record=True)#monitor voltage to view spikes and g_ampa for conductance
    counter_out = SpikeMonitor(output_layer, record=True)#count the spikes in the output layer
    counter_hidden = SpikeMonitor(hidden_layer,record=True)#count the spikes in the hidden layer
    counter_in = SpikeMonitor(input_layer,record=True)#count the spikes in the input layer
#initalize the Network object with our Brian2 objects, this prevents errors that occur when just using a MagicNetwork
    net = Network(mon,counter_out,counter_hidden,counter_in, input_layer, hidden_layer, output_layer, S_ij[:], S_jk[:])
    
    #keep track of the total number of spikes of each layer across iterations
    #necesary because our spike monitors track the running total, but for our math, we need iteration-totals
    prev_full_count_in = 0
    prev_full_count_hidden = 0
    prev_full_count_out = 0
    
    #keep track of the total number of spikes of each neuron of each layer for same reasons as above
    prev_count_in = [0]*len(input_layer)
    prev_count_hidden = [0]*len(hidden_layer)
    prev_count_out = [0]*len(output_layer)
    
    #iterate through runs and weight updates for the number of examples/hands (len(cards))
    for numHand in range(len(cards)):
        
        #set all to 0*nA initially to allow reset of voltage
        input_layer.I = 0*nA
        hidden_layer.I = 0*nA
        output_layer.I=0*nA
        net.run(50*ms,report='text')
        
        #run only the cards (five in range 0-52) included in the hand at 8*nA to generate spikes for these neurons
        for i in range(len(cards[numHand])):
            input_layer.I[cards[numHand][i]] = 8*nA
        net.run(7*ms, report='text')
    	#run again at 0*nA to allow time to reset
        input_layer.I = 0*nA
        hidden_layer.I = 0*nA
        output_layer.I=0*nA
        net.run(30.0*ms)
        
        #keep track of spike counts of each neuron for use in back-propagation math
        count_in = []
        count_hidden = []
        count_out = []
        
        #keep track of spike count proportions of each neuron for use in back-propagation math
        #these values take the spike count of the individual neuron divided by total spikes in the layer - range [0,1)
        #gets this value by taking the running total minus the previous running total
        prop_count_in = []
        prop_count_hidden = []
        prop_count_out = []
    
        #get total number of spikes in each layer FOR THIS ITERATION
        #gets this value by taking the running total minus the previous running total
        full_count_in = counter_in.num_spikes -prev_full_count_in
        full_count_hidden = counter_hidden.num_spikes - prev_full_count_hidden
        full_count_out = counter_out.num_spikes - prev_full_count_out
        
        #gets arrays for the count of each neuron in layer, as well as the proportion of spikes a neuron has with its layer
        for ci in range(len(input_layer)):
            count_in.append(counter_in.count[ci]-prev_count_in[ci])
            prev_count_in[ci]+=count_in[ci] #update running count of neuron spikes
            prop_count_in.append(float(count_in[ci])/full_count_in)
            #nested if loops here mean we don't need 3 for loops
            #the math still checks out because the spikes are counted beforehand during run()
            if ci<len(hidden_layer):
                count_hidden.append(counter_hidden.count[ci]-prev_count_hidden[ci])
                prev_count_hidden[ci]+=count_hidden[ci]#update running count of neuron spikes
                prop_count_hidden.append(float(count_hidden[ci])/full_count_hidden)
            if ci<len(output_layer):
                count_out.append(counter_out.count[ci]-prev_count_out[ci])
                prev_count_out[ci]+=count_out[ci] #update running count of neuron spikes
                prop_count_out.append(float(count_out[ci])/full_count_out)
        #get numpy array representation so we can do matrix math        
        prop_count_in = np.asarray(prop_count_in)
        prop_count_hidden = np.asarray(prop_count_hidden)
        prop_count_out = np.asarray(prop_count_out)   
        
		#update the running count of total spikes
        prev_full_count_in += full_count_in
        prev_full_count_hidden += full_count_hidden
        prev_full_count_out += full_count_out

        prop_out_preSig = 2*(prop_count_out)-1
		
		#sigmoid activation on proportion of spikes
        yk = sigmoid(prop_out_preSig,prime=False)

		#error in expected versus actual 'e_k'
        # where actual is the proportion of spikes of that neuron
        err = label[numHand] - yk
        #delta_k is the error gradient
        #we dont take the sigmoid of (prop_count_out) beacuse we get our neuron 'value' from spike counter
        delta_k = sigmoid(yk,prime=True)*err
        
        prop_hidden_preSig = 2*(prop_count_hidden)-1
        yj = sigmoid(prop_hidden_preSig,prime=False)
        
		#get the change in weights for the jk layer
        delta_Wjk = []
        for j in range(len(hidden_layer)):
            for k in range(len(output_layer)):        
                delta_Wjk.append(alpha*yj[j]*delta_k[k])
    	#update the Synapse weights between layer j&k
        for k in range(len(S_jk)):
            S_jk[k].w_jk +=delta_Wjk[k]
        
		#interm is the e_j error in the middle layer
        interm = np.zeros(len(hidden_layer))
        for j in range(len(hidden_layer)):
            for k in range(len(output_layer)):
                interm[j] +=delta_k[k]*S_jk[k+k*j].w_jk
        #delta_k is the error gradient
        #we dont take the sigmoid of (prop_count_out) beacuse we get our neuron 'value' from spike counter
        delta_j = sigmoid(prop_count_hidden,prime=True)*interm
        
        #get the change in weights for the ij layer
        delta_Wij = []
        for i in range(len(input_layer)):
            for j in range(len(hidden_layer)):
                delta_Wij.append(alpha*prop_count_in[i]*delta_j[j])
        #update the Synapse weights between layer i&j
        for j in range(len(S_ij)):
            S_ij[j].w_ij +=delta_Wij[j]
        
    
 		## print all the math we just did

        #print("deltaij: ", delta_j)
        #print("deltaWij:",delta_Wij[0])
    
        #print('delta_j: ', delta_j)
        #print("new weights ij", S_ij[0].w_ij)
        #print("counts in:", count_in)
        #print("counts ou:", count_out)
    
        #print("full count in", full_count_in)
        print ("full count out",full_count_out)
        #print("prop out", prop_count_out)
        #print("prop hidden", prop_count_hidden)
        
        #print the trial and the error
        #ideally these should all be close to 0
        #becuase the one hot label thats one should have all the spikes (proportion=1)
        #and all the 0's from label should have proportion=0
        print('trial: ',numHand,'error: ',err)
        errorList.append(err)
        errorSum = np.sqrt(sum(np.square(err))/10)
        errorSumList.append(errorSum)
        print('RMSE:',errorSum,'SquareList:  ', err)
        
        
	
    iters = []
    for yuh in range(len(cards)):
        iters.append(yuh)
    
	#plot the spikes in the output layer neurons
    #currently only plots first 3 (high card, one pair, two pair)
    #use this as an example for if you want to plot your own things

	plt.figure()
    plt.subplot(4,1,1)
    plt.plot(mon.t/ms, mon.v[0]/mV)
    plt.title('Hand 0')
    plt.subplot(4,1,2)
    plt.plot(mon.t/ms, mon.v[1]/mV,'r')
    plt.title('Hand 1')
    plt.subplot(4,1,3)
    plt.plot(mon.t/ms, mon.v[2]/mV,'g')
    plt.title('Hand 2')
    plt.subplot(4,1,4)
    plt.plot(iters,errorSumList)
    plt.title('RMSE')
    plt.show()
    
	#once training is done, save all the weights into a list of weights
    #each value in the list will be of type numpy.ndarray
    #this is so that we can easily initialize a network with these weights for the Synapse objects
    weights_ij=[]
    for wij_count in range(len(S_ij)):
        weights_ij.append(S_ij[wij_count].w_ij)
    
    weights_jk=[]
    for wjk_count in range(len(S_jk)):
        weights_jk.append(S_jk[wjk_count].w_jk)
    return weights_ij, weights_jk
