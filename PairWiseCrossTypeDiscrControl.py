#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:39:38 2017

@author: frederictheunissen
"""
#%% Dependencies
import pandas
import numpy as np
import matplotlib.pyplot as plt
# import string as str
from soundsig.discriminate import discriminatePlot
from scipy.stats.mstats import zscore
from scipy.stats import binom
from statsmodels.stats.proportion import proportion_confint

#%% Stores 
inputTable = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/vocParamTable.h5'
outputTable = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossControlResults.h5'
outputGroupedTable = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossControlGroupedResults.h5'
fileExcelTable = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossDiscControlResults.xls'
figdir = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/Figures Voice'

#%% Read data
vocSelTable = pandas.read_hdf(inputTable)

#%% List of call types and of birnames
callTypes = np.unique(vocSelTable['calltype'])
birdNames = np.unique(vocSelTable['Bird'])

# Color code for bird ID
birdColor = {}
for birdId in birdNames:
    birdColor[birdId] = np.random.rand(3)
    
# Select the data that will be used in the classifier
y = np.array(vocSelTable['Bird'])

# Divide features into Fundamental, Spectral and Temporal

XfundNames = np.hstack(('sal', 'fund','maxfund','minfund','cvfund'))
Xfund = [vocSelTable.loc[:,col] for col in XfundNames ]
Xfund = np.stack(Xfund, axis=1)

XspectNames = np.hstack(('meanS', 'stdS', 'skewS', 'kurtS', 'entS', 'q1', 'q2', 'q3'))
Xspect = [vocSelTable.loc[:,col] for col in XspectNames ]
Xspect = np.stack(Xspect, axis=1)

XtempNames = np.hstack(('meanT', 'stdT', 'skewT', 'kurtT', 'entT'))
Xtemp = [vocSelTable.loc[:,col] for col in XtempNames ]
Xtemp = np.stack(Xtemp, axis=1)

   
Xall = np.hstack((Xfund, Xspect, Xtemp))
XallNames = np.hstack((XfundNames, XspectNames, XtempNames))

# Z score all data that is not nan and make corresponding arrays for bird id and call type
nonanInd = (np.sum(np.isnan(Xall), axis = 1) == 0)

Xfundz = zscore(Xfund[nonanInd], axis=0)
Xspectz = zscore(Xspect[nonanInd], axis=0)
Xtempz = zscore(Xtemp[nonanInd], axis=0)
Xallz = zscore(Xall[nonanInd], axis=0)

birdz = y[nonanInd]
birdSex = np.array(vocSelTable['Sex'])[nonanInd]
callTypez = np.asarray(vocSelTable['calltype'])[nonanInd]

# Generate a unique randome color for each bird
cValBirdAll = []
for birdId in birdz:
    cValBirdAll.append(birdColor[birdId])
cValBirdAll = np.asarray(cValBirdAll)

#%% Perform the pair-wise discrimination for each call type (triple loop)


# Loop through all birds:
#for ctype in callTypes:
#    goodInd = callTypez == ctype
#    goodInd = goodInd.nonzero()
#    nsamples = np.size(goodInd)
#    birds = np.unique(birdz[goodInd])
#    nbirds = birds.size
#    print ('Performing Voice Discrimination for %s with %d samples from %d birds' % (ctype, nsamples, nbirds))
#    if (nbirds < 2) :
#        print ('Warning: Insuficient number of birds to perform discrimination')
#    else:

MINCOUNT = 10
MINCLASSCOUNT = 5
# Loop through all birds:        
birds = np.unique(birdz)
nbirds = birds.size   
goodPairs = []  # List of info for good Pairs    
for i1 in range(nbirds):
    bird1 = birds[i1]
    for i2 in range(i1+1, nbirds):
        bird2 = birds[i2]
                
        # The index that have each of these birds
        birdPairInd = (birdz == bird1) | (birdz == bird2)

                
        # These are all calls types for this bird pair. 
        birdzPair = birdz[birdPairInd]
        callTypezPair = callTypez[birdPairInd]
        callTypeUnique = np.unique(callTypezPair)
                
        # Find the minimum value of calls in each group

        classCount = 0
        for ctype in callTypeUnique:
            ctypeIndBird1 = (callTypezPair == ctype) & (birdzPair == bird1)
            ctypeIndBird2 = (callTypezPair == ctype) & (birdzPair == bird2)
            countType1 = np.sum(ctypeIndBird1)
            countType2 = np.sum(ctypeIndBird2)
            if countType1 < MINCOUNT or countType2 < MINCOUNT:
                continue
            classCount += 1
         
        if classCount >= MINCLASSCOUNT : 
            minCalls = np.sum(birdPairInd)
            minCallType = None
            print ('Bird Pair %s %s:'%(bird1, bird2) )
            for ctype in callTypeUnique:
                ctypeIndBird1 = (callTypezPair == ctype) & (birdzPair == bird1)
                ctypeIndBird2 = (callTypezPair == ctype) & (birdzPair == bird2)
                countType1 = np.sum(ctypeIndBird1)
                countType2 = np.sum(ctypeIndBird2)
                if countType1 < MINCOUNT or countType2 < MINCOUNT:
                    continue          
                print ('\t Call %s # (%d, %d)' % (ctype, countType1, countType2)  )
                if countType1 < minCalls :
                    minCalls = countType1
                    minCallType = ctype
                if countType2 < minCalls :
                    minCalls = countType2
                    minCallType = ctype
            
            print('Minimum number of class found for %s = %s' % (minCalls, minCallType))
            
            # Sex labeling
            birdSexPair = birdSex[birdPairInd]
            sexPairs = np.unique(birdSexPair)
            if len(sexPairs) == 1 :
                if 'M' in sexPairs :
                    sexLabel = 'M'
                elif 'F' in sexPairs :
                    sexLabel = 'F'
                else:
                    sexLabel = 'U'
            else:
                if ('M' in sexPairs) and ('F' in sexPairs) :
                    sexLabel = 'X'
                else:
                    sexLabel = 'U'
            
            goodPairs.append({'bird1': bird1, 'bird2' : bird2, 'minCalls': minCalls, 'sexLabel': sexLabel})
            
           

print('Found %d pairs that satisfy requirement' % len(goodPairs))

# Generate a training set for that bird pair
for gp in goodPairs:
    print('Pair %s-%s: %d min calls' % (gp['bird1'], gp['bird2'], gp['minCalls']))
 
#%% Preforming the LDA

ldaYes = []
qdaYes = []
rfYes = [] 
cvCount = []
ldaProb = []
qdaProb = []
rfProb = []
nClasses = []
callType = []
Features = []
Weights = []
SexPair = []
BirdPair = []
testType = []

    
# The loop through bird pairs that have passed the test in the previous section
for gp in goodPairs:    
    birdPairInd = (birdz == gp['bird1']) | (birdz == gp['bird2'])
                
    # These are data for this bird pair. 
    XfundzPair = Xfundz[birdPairInd]
    XspectzPair = Xspectz[birdPairInd]
    XtempzPair = Xtempz[birdPairInd] 
    XallzPair = Xallz[birdPairInd]

    birdzPair = birdz[birdPairInd]  
    cValBirdPair = cValBirdAll[birdPairInd]
    callTypezPair = callTypez[birdPairInd]
    callTypeUnique = np.unique(callTypezPair)
    
    featureList = [{'name' : '18 AF', 'X' : XallzPair}, 
                   {'name' : 'Fund AF', 'X' : XfundzPair},
                   {'name' : 'Spect AF', 'X' : XspectzPair},
                   {'name' : 'Temp AF', 'X' : XtempzPair} ]

    for ctypeTest in callTypeUnique:
        ctypeIndBird1 = (callTypezPair == ctypeTest) & (birdzPair == gp['bird1'])
        ctypeIndBird2 = (callTypezPair == ctypeTest) & (birdzPair == gp['bird2'])
        countType1 = np.sum(ctypeIndBird1)
        countType2 = np.sum(ctypeIndBird2)
        if countType1 < MINCOUNT or countType2 < MINCOUNT:
            continue
        
        ctypeIndBird1 = np.nonzero(ctypeIndBird1)[0]
        ctypeIndBird2 = np.nonzero(ctypeIndBird2)[0]
        nall1 = len(ctypeIndBird1)
        nall2 = len(ctypeIndBird2)
        ntrain1 = np.int(nall1*0.8)
        ntrain2 = np.int(nall2*0.8)
        
        trainInd = np.hstack((ctypeIndBird1[0:ntrain1], ctypeIndBird2[0:ntrain2]))
        testInd = np.hstack((ctypeIndBird1[ntrain1:], ctypeIndBird2[ntrain2:]))
        allInd = np.hstack((trainInd,testInd))
                 
        for fl in featureList:
            # Perform pair-wise discrimination for within call first
            ldaY, qdaY, rfY, cvC, ldaP, qdaP, rfP, nC, ldaweights = discriminatePlot(fl['X'][allInd], birdzPair[allInd], cValBirdPair[allInd], 
                                                                                 titleStr='Caller %s (%s vs %s) 18 AF' % (ctypeTest, gp['bird1'], gp['bird2']), 
                                                                                 figdir = figdir, Xcolname = XallNames)
        
            if ldaY == -1:
                print ('Error: Failure in discriminate Plot')
                          
            # Store data 
            ldaYes.append(ldaY)
            ldaProb.append(ldaP)
            qdaYes.append(qdaY)
            qdaProb.append(qdaP)
            rfYes.append(rfY) 
            rfProb.append(rfP)
            cvCount.append(cvC)
            nClasses.append(nC)
            Features.append(fl['name'])
            Weights.append(abs(ldaweights).mean(axis=0))
            SexPair.append(gp['sexLabel'])
            BirdPair.append((gp['bird1'], gp['bird2']))
            callType.append(ctypeTest)
            testType.append('Caller')
            
            # Perform pair-wise discrimination for across calls
            ntrain = len(trainInd)
            ntest = len(testInd)
            testIndTF = np.asarray([False] * ntrain + [True] * ntest)
            
            # Perform pair-wise discrimination for within call first
            ldaY, qdaY, rfY, cvC, ldaP, qdaP, rfP, nC, ldaweights = discriminatePlot(fl['X'][allInd], birdzPair[allInd], cValBirdPair[allInd], 
                                                                                 titleStr='Voice %s (%s vs %s) 18 AF' % (ctypeTest, gp['bird1'], gp['bird2']), 
                                                                                 figdir = figdir, Xcolname = XallNames, testInd = testIndTF)
            # Store data 
            ldaYes.append(ldaY)
            ldaProb.append(ldaP)
            qdaYes.append(qdaY)
            qdaProb.append(qdaP)
            rfYes.append(rfY) 
            rfProb.append(rfP)
            cvCount.append(cvC)
            nClasses.append(nC)
            Features.append(fl['name'])
            Weights.append(abs(ldaweights).mean(axis=0))
            SexPair.append(gp['sexLabel'])
            BirdPair.append((gp['bird1'], gp['bird2']))
            callType.append(ctypeTest)
            testType.append('Voice')
            
d = {'Call': np.array(callType),
     'TestType' : np.array(testType),
     'BirdPair': BirdPair, 
     'SexPair': SexPair,
     'Features': np.array(Features), 
     'LDAYes' : np.array(ldaYes), 
     'QDAYes': np.array(qdaYes),
     'RFYes': np.array(rfYes), 
     'Count': np.array(cvCount),
     'nClasses' : np.array(nClasses),
     'LDA_P' : np.array(ldaProb),
     'QDA_P': np.array(qdaProb),
     'RF_P': np.array(rfProb),
     'Weights': Weights
     }
 
# Store data in Pandas Data Frame.    
resultsDataFrame = pandas.DataFrame(data=d)
                

#%% Generate aggregate Data Frames.  

### Warning this code might need an update. Ignores pair-wise discrimination.

# Find the index that are unique to each Type
resultsGroupedAll = resultsDataFrame.groupby(['Call', 'TestType', 'Features'])
resultsGroupedSex = resultsDataFrame.groupby(['Call', 'TestType', 'Features', 'SexPair'])

# # Make an Aggregate Pandas Data Frame by hand.  Pandas aggregate did not work well with mixed data.
numberOfRows = len(resultsGroupedAll.indices) + len(resultsGroupedSex.indices)
resultsAgg = pandas.DataFrame(index=np.arange(0, numberOfRows), 
            columns=('Call', 'TestType', 'Features', 'SexPair', 'NPairs', 'TestCount',
                     'LDA', 'LDA_YES', 'LDA_P', 'QDA', 'QDA_YES', 'QDA_P', 
                     'RF', 'RF_YES', 'RF_P', 'Weights', 'Weights_SE') )

i = 0
for key in resultsGroupedAll.indices:
       
    # Extract Call Type from key
    resultsAgg.iloc[i,0] = key[0]   # CallType
    
    # Extract Test Type from key
    resultsAgg.iloc[i,1] = key[1]   # Test Type
    
    # Extract Features from key
    resultsAgg.iloc[i,2] = key[2]
    
    # Sex Pair Code
    resultsAgg.iloc[i,3] = 'A'   # A for all: M, F, X and U

    # Number of pair-wise comparisons
    n = len(resultsGroupedAll.indices[key])
    resultsAgg.iloc[i,4] = n # NPairs
    
    # Number of trials used in cross-validation
    resultsAgg.iloc[i,5] = resultsDataFrame.Count.iloc[resultsGroupedAll.indices[key]].sum()
    
    # LDA_YES and LDA_P
    resultsAgg.iloc[i,7] = resultsDataFrame.LDAYes.iloc[resultsGroupedAll.indices[key]].sum() 
    resultsAgg.iloc[i,6] =  100.0*resultsAgg.iloc[i,7]/resultsAgg.iloc[i,5]
    ldaP = 0
    p = 0.5   # Assuming 2 classes here
    for k in range(resultsAgg.iloc[i,7], resultsAgg.iloc[i,5]+1):
        ldaP += binom.pmf(k, resultsAgg.iloc[i,5], p)
    resultsAgg.iloc[i,8] = ldaP
    
    # QDA_YES and QDA_P
    resultsAgg.iloc[i,10] = resultsDataFrame.QDAYes.iloc[resultsGroupedAll.indices[key]].sum() 
    resultsAgg.iloc[i,9] =  100.0*resultsAgg.iloc[i,10]/resultsAgg.iloc[i,5]
    qdaP = 0
    p = 0.5   # Assuming 2 classes here
    for k in range(resultsAgg.iloc[i,10], resultsAgg.iloc[i,5]+1):
        qdaP += binom.pmf(k, resultsAgg.iloc[i,5], p)
    resultsAgg.iloc[i,11] = qdaP
       
    # RF and RF_SE
    resultsAgg.iloc[i,13] = resultsDataFrame.RFYes.iloc[resultsGroupedAll.indices[key]].sum()
    resultsAgg.iloc[i,12] =  100.0*resultsAgg.iloc[i,13]/resultsAgg.iloc[i,5]
    rfP = 0
    p = 0.5   # Assuming 2 classes here
    for k in range(resultsAgg.iloc[i,13], resultsAgg.iloc[i,5]+1):
        rfP += binom.pmf(k, resultsAgg.iloc[i,5], p)
    resultsAgg.iloc[i,14] = rfP
                   
    # Average Weights and SE
    WeightsAll = np.vstack(resultsDataFrame.Weights.iloc[resultsGroupedAll.indices[key]])
    countAll = np.array(resultsDataFrame.Count.iloc[resultsGroupedAll.indices[key]], ndmin=2 )
    WeightsMean = np.dot(countAll, WeightsAll)/np.sum(countAll)
    
    WeightsSE = np.sqrt(np.dot(countAll, (WeightsAll-WeightsMean)**2)/(np.sum(countAll)*n))
    resultsAgg.iloc[i,15] = WeightsMean
    resultsAgg.iloc[i,16] = WeightsSE               
                   
    i += 1
    
for key in resultsGroupedSex.indices:
       
    # Extract Call Type from key
    resultsAgg.iloc[i,0] = key[0]   # CallType
    
    # Extract Test type from key
    resultsAgg.iloc[i,1] = key[1]
    
    # Extract Features from key
    resultsAgg.iloc[i,2] = key[2]
    
    # Sex Pair Code
    resultsAgg.iloc[i,3] = key[3]   # A M, F, X or U

    # Number of pair-wise comparisons
    n = len(resultsGroupedSex.indices[key])
    resultsAgg.iloc[i,4] = n # NPairs
    
    # Number of trials used in cross-validation
    resultsAgg.iloc[i,5] = resultsDataFrame.Count.iloc[resultsGroupedSex.indices[key]].sum()
    
    # LDA_YES and LDA_P
    resultsAgg.iloc[i,7] = resultsDataFrame.LDAYes.iloc[resultsGroupedSex.indices[key]].sum() 
    resultsAgg.iloc[i,6] =  100.0*resultsAgg.iloc[i,7]/resultsAgg.iloc[i,5]
    ldaP = 0
    p = 0.5   # Assuming 2 classes here
    for k in range(resultsAgg.iloc[i,7], resultsAgg.iloc[i,5]+1):
        ldaP += binom.pmf(k, resultsAgg.iloc[i,5], p)
    resultsAgg.iloc[i,8] = ldaP
    
    # QDA_YES and QDA_P
    resultsAgg.iloc[i,10] = resultsDataFrame.QDAYes.iloc[resultsGroupedSex.indices[key]].sum() 
    resultsAgg.iloc[i,9] =  100.0*resultsAgg.iloc[i,10]/resultsAgg.iloc[i,5]
    qdaP = 0
    p = 0.5   # Assuming 2 classes here
    for k in range(resultsAgg.iloc[i,10], resultsAgg.iloc[i,5]+1):
        qdaP += binom.pmf(k, resultsAgg.iloc[i,5], p)
    resultsAgg.iloc[i,11] = qdaP
       
    # RF and RF_SE
    resultsAgg.iloc[i,13] = resultsDataFrame.RFYes.iloc[resultsGroupedSex.indices[key]].sum()
    resultsAgg.iloc[i,12] =  100.0*resultsAgg.iloc[i,12]/resultsAgg.iloc[i,5]
    rfP = 0
    p = 0.5   # Assuming 2 classes here
    for k in range(resultsAgg.iloc[i,13], resultsAgg.iloc[i,5]+1):
        rfP += binom.pmf(k, resultsAgg.iloc[i,5], p)
    resultsAgg.iloc[i,14] = rfP
                   
    # Average Weights and SE
    WeightsAll = np.vstack(resultsDataFrame.Weights.iloc[resultsGroupedSex.indices[key]])
    countAll = np.array(resultsDataFrame.Count.iloc[resultsGroupedSex.indices[key]], ndmin=2 )
    WeightsMean = np.dot(countAll, WeightsAll)/np.sum(countAll)
    
    WeightsSE = np.sqrt(np.dot(countAll, (WeightsAll-WeightsMean)**2)/(np.sum(countAll)*n))
    resultsAgg.iloc[i,15] = WeightsMean
    resultsAgg.iloc[i,16] = WeightsSE                 
                   
    i += 1

# Print average performances of Classifiers
print('Weighted Average')
print('LDA:', 100.0*resultsAgg.LDA_YES.sum()/resultsAgg.TestCount.sum(), ' QDA:', 100.0*resultsAgg.QDA_YES.sum()/resultsAgg.TestCount.sum(), ' RF:', 100.0*resultsAgg.RF_YES.sum()/resultsAgg.TestCount.sum() )

print('Average (per call - across all')
print('LDA:', resultsAgg.LDA.mean(), ' QDA:', resultsAgg.QDA.mean(), ' RF:', resultsAgg.RF.mean() )

#%% Save Data
resultsDataFrame.to_hdf(outputTable, 'resultsDataFrame', mode='w')
resultsAgg.to_hdf(outputGroupedTable, 'resultsAgg', mode='w')

# Write the results to Excel without the Weights
resultsDataFrame.to_excel(fileExcelTable, columns = resultsDataFrame.columns[0:12])

#%% Read Data - one could start here.
resultsDataFrame = pandas.read_hdf(outputTable)
resultsAgg = pandas.read_hdf(outputGroupedTable)

#%% Make a box plot to summarize results

# Select rows for all sex combinations and all Features.
indCallerAll = [index for index, row in resultsAgg.iterrows() if ( (row.SexPair == "A") and (row.Features == '18 AF') and (row.TestType == 'Caller'))]
# indVoiceAll = [index for index, row in resultsAgg.iterrows() if ( (row.SexPair == "A") and (row.Features == '18 AF') and (row.TestType == 'Voice'))]

resultsCallerAll = resultsAgg.loc[indCallerAll]

# Sort data by descending performance for LDA discrimination
resultsCallerAll = resultsCallerAll.sort_values(by='LDA', ascending = False) 
sortedCallTypes = resultsCallerAll.Call   # Sorted Call Types

# V for Voice
indVoiceAll = []
xvalsV = []
xvals = -1
oldlen = 0
for lbl in sortedCallTypes:
    xvals += 1
    indVoiceAll += [index for index, row in resultsAgg.iterrows() if ( (row.SexPair == 'A') and (row.Call == lbl) and (row.Features == '18 AF') and (row.TestType == 'Voice')) ]
    newlen = len(indVoiceAll)
    if (newlen != oldlen):   # This code is for the x axis for the plot to skip if there is no data.
        xvalsV += [xvals]
        oldlen = newlen
resultsVoiceAll = resultsAgg.loc[indVoiceAll]

# Color code for call type
callColor = {'Be': (0/255.0, 230/255.0, 255/255.0), 'LT': (0/255.0, 95/255.0, 255/255.0), 'Tu': (255/255.0, 200/255.0, 65/255.0), 'Th': (255/255.0, 150/255.0, 40/255.0), 
             'Di': (255/255.0, 105/255.0, 15/255.0), 'Ag': (255/255.0, 0/255.0, 0/255.0), 'Wh': (255/255.0, 180/255.0, 255/255.0), 'Ne': (255/255.0, 100/255.0, 255/255.0),
             'Te': (140/255.0, 100/255.0, 185/255.0), 'DC': (100/255.0, 50/255.0, 200/255.0), 'So': (0/255.0, 0/255.0, 0/255.0)}


plt.figure()
xvals = np.arange(len(indCallerAll))
width = 0.75          # the width of the bars
numbars = 2           # Number of bars per group
 
plt.subplot(131)
yerr = np.asarray([ proportion_confint(count, nobs) for count, nobs in zip(resultsCallerAll.LDA_YES, resultsCallerAll.TestCount)])
b1=plt.bar(xvals*numbars, resultsCallerAll.LDA, width, color='k', yerr=yerr.T)

yerr = np.asarray([ proportion_confint(count, nobs) for count, nobs in zip(resultsVoiceAll.LDA_YES, resultsVoiceAll.TestCount)])
b2=plt.bar(np.array(xvalsV)*numbars + width, resultsVoiceAll.LDA, width, color='b', yerr=yerr.T)                      
                    
plt.xticks(numbars*xvals + width*numbars/2., sortedCallTypes)
plt.legend((b1[0], b2[0]), ('Within CV', 'Across CV'))
plt.title('Within Call Type vs Across Call Type')
plt.ylabel('LDA Performance % ')
plt.xlabel('Call Type')
myAxis = plt.axis()
myAxis = (myAxis[0], myAxis[1], 40.0, myAxis[3])
plt.axis(myAxis)

plt.subplot(132)

yerr = np.asarray([ proportion_confint(count, nobs) for count, nobs in zip(resultsCallerAll.QDA_YES, resultsCallerAll.TestCount)])
b1=plt.bar(xvals*numbars, resultsCallerAll.QDA, width, color='k', yerr=yerr.T)

yerr = np.asarray([ proportion_confint(count, nobs) for count, nobs in zip(resultsVoiceAll.QDA_YES, resultsVoiceAll.TestCount)])
b2=plt.bar(np.array(xvalsV)*numbars + width, resultsVoiceAll.QDA, width, color='b', yerr=yerr.T)                      
                    
plt.xticks(numbars*xvals + width*numbars/2., sortedCallTypes)
plt.legend((b1[0], b2[0]), ('Within CV', 'Across CV'))
plt.title('Within Call Type vs Across Call Type')
plt.ylabel('QDA Performance % ')
plt.xlabel('Call Type')
myAxis = plt.axis()
myAxis = (myAxis[0], myAxis[1], 40.0, myAxis[3])
plt.axis(myAxis)

plt.subplot(133)

yerr = np.asarray([ proportion_confint(count, nobs) for count, nobs in zip(resultsCallerAll.RF_YES, resultsCallerAll.TestCount)])
b1=plt.bar(xvals*numbars, resultsCallerAll.RF, width, color='k', yerr=yerr.T)

yerr = np.asarray([ proportion_confint(count, nobs) for count, nobs in zip(resultsVoiceAll.RF_YES, resultsVoiceAll.TestCount)])
b2=plt.bar(np.array(xvalsV)*numbars + width, resultsVoiceAll.RF, width, color='b', yerr=yerr.T )                     
                    

plt.xticks(numbars*xvals + width*numbars/2., sortedCallTypes)
plt.legend((b1[0], b2[0]), ('Within CV', 'Across CV'))
plt.title('Within Call Type vs Across Call Type')
plt.ylabel('RF Performance % ')
plt.xlabel('Call Type')
myAxis = plt.axis()
myAxis = (myAxis[0], myAxis[1], 40.0, myAxis[3])
plt.axis(myAxis)


plt.show()

#%% Make a nicer bar graph
hatch = {'C': '', 'V': ''}
testKeys = ['C', 'V']
results = {'C' : resultsCallerAll.QDA, 'V' : resultsVoiceAll.QDA }
yerrCaller = np.asarray([ proportion_confint(count, nobs) for count, nobs in zip(resultsCallerAll.QDA_YES, resultsCallerAll.TestCount)])
yerrVoice = np.asarray([ proportion_confint(count, nobs) for count, nobs in zip(resultsVoiceAll.QDA_YES, resultsVoiceAll.TestCount)])
yerrKeyed = {'C': yerrCaller, 'V': yerrVoice }

plt.figure(figsize=(4,3))    # Standard plotting figures are 4 inches wide by 3 inches high
ngrps = len(testKeys)+1
width = 0.75          # the width of the bars
bottom = 0.5          # The bottom value of the bars
bp = {}

for i, callkey in enumerate(sortedCallTypes):
    for j, testkey in enumerate(testKeys):
        barval = (results[testkey].iloc[i])*0.01
        yerrval = np.vstack((barval-yerrKeyed[testkey][i][0], yerrKeyed[testkey][i][1]-barval))
        print(barval, yerrval)
        bp[callkey+testkey] = plt.bar(i*ngrps+j, barval-bottom, width = 0.75, color=callColor[callkey], 
            bottom = bottom, hatch = hatch[testkey],
            yerr=yerrval, ecolor = (0.0, 0.0, 0.0), error_kw = {"elinewidth": 2, "capthick" : 2}, capsize = 4)
        plt.text(i*ngrps+j, 0.44, testkey, fontsize=10)

# The location of x tick marks
xvals = np.asarray(range(len(sortedCallTypes)))*ngrps + (len(testKeys)-1)/2.0 + width/2.
plt.xticks(xvals, sortedCallTypes)
# plt.legend((bp['DCF'], bp['DCM'], bp['DCX']), ('F-F', 'M-M', 'F-M'))
# plt.title('Classification for Caller ID per Sex')
plt.ylabel('% QDA', fontsize = 10, weight="bold")
plt.xlabel('Call Type', fontsize = 10, weight="bold")

# plt.yscale('logit')
plt.xlim([-1, 24])
plt.ylim([0.4, 0.98])
plt.plot([-1, 24], [0.5, 0.5], color = 'k', LineWidth = 2 )

#%% Make a scatter plot

# Select rows for all sex combinations and all Features.
indCallerAll = [index for index, row in resultsDataFrame.iterrows() if ((row.Features == '18 AF') and (row.TestType == 'Caller'))]

plt.figure(figsize=(4,3))

for i in indCallerAll:
    rowCaller = resultsDataFrame.iloc[i]
    
    # Find correspond row for voice
    rowVoice = None
    for j, row in resultsDataFrame.iterrows():
        if (row.BirdPair == rowCaller.BirdPair) and (row.Features == rowCaller.Features) and (row.Call == rowCaller.Call) and (row.TestType == 'Voice') :
            rowVoice = row
            break
    
    # Plot the point with the right color.
    if rowVoice is None:
        print('Error: could not find voice data for birds %s, call %s and features %s' % (rowCaller.Birdpair, rowCaller.Caller, rowCaller.Features) )
    else:
        plt.plot(float(rowCaller.LDAYes)/rowCaller.Count,float(rowVoice.LDAYes)/rowVoice.Count, 'o', color = callColor[rowCaller.Call] )
    
plt.plot([0.5, 1.0], [0.5, 1.0], color = 'k')
plt.xlabel('Caller Discrimination')
plt.ylabel('Voice Discrimination')
    
    
    