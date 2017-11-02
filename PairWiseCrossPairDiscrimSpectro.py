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
inputTable = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/vocSpectroTable.h5'
outputTable = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossPairDiscrimSpectroResults.h5'
outputGroupedTable = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossPairDiscrimSpectroGroupedResults.h5'
fileExcelTable = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossPairDiscrimSpectroResults.xls'
figdir = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/Figures Voice'

#%% Read data
vocSelTable = pandas.read_hdf(inputTable)

#%% List of call types and of birnames
callTypes = np.unique(vocSelTable['Calltype'])
birdNames = np.unique(vocSelTable['Bird'])

# Color code for bird ID
birdColor = {}
for birdId in birdNames:
    birdColor[birdId] = np.random.rand(3)
    
# Select the data that will be used in the classifier
y = np.array(vocSelTable['Bird'])

# Divide features into Fundamental, Spectral and Temporal

# Feature space is the spectrogram
   
Xall = np.stack(vocSelTable.loc[: ,'Spectro'], axis=0)
XallNames = 'Spectro'

# Z score all data that is not nan and make corresponding arrays for bird id and call type
nonanInd = (np.sum(np.isnan(Xall), axis = 1) == 0)

Xallz = zscore(Xall[nonanInd], axis=0)


birdz = y[nonanInd]
birdSex = np.array(vocSelTable['Sex'])[nonanInd]
callTypez = np.asarray(vocSelTable['Calltype'])[nonanInd]

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
MINCLASSCOUNT = 2
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
callTrain = []
callTest = []
Features = []
Weights = []
SexPair = []
BirdPair = []


    
# The loop through bird pairs that have passed the test in the previous section
for gp in goodPairs:    
    birdPairInd = (birdz == gp['bird1']) | (birdz == gp['bird2'])
                
    # These are data for this bird pair.  
    XallzPair = Xallz[birdPairInd]

    birdzPair = birdz[birdPairInd]  
    cValBirdPair = cValBirdAll[birdPairInd]
    callTypezPair = callTypez[birdPairInd]
    callTypeUnique = np.unique(callTypezPair)
    
    featureList = [{'name' : 'Spectro', 'X' : XallzPair}] 

    for ctypeTest in callTypeUnique:
        ctypeIndBird1 = (callTypezPair == ctypeTest) & (birdzPair == gp['bird1'])
        ctypeIndBird2 = (callTypezPair == ctypeTest) & (birdzPair == gp['bird2'])
        countType1 = np.sum(ctypeIndBird1)
        countType2 = np.sum(ctypeIndBird2)
        if countType1 < MINCOUNT or countType2 < MINCOUNT:
            continue
        testInd = np.asarray(callTypezPair == ctypeTest).nonzero()[0]
        
        # First Perform pair-wise discrimination for within call
        for fl in featureList:               
            ldaY, qdaY, rfY, cvC, ldaP, qdaP, rfP, nC, ldaweights = discriminatePlot(fl['X'][testInd], birdzPair[testInd], cValBirdPair[testInd], 
                                                                                 titleStr='Caller %s (%s vs %s) Spectro' % (ctypeTest, gp['bird1'], gp['bird2']), 
                                                                                 figdir = figdir)
       
            if ldaY == -1:
                print ('Error: Failure in discriminate Plot')
                continue
                          
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
            callTest.append(ctypeTest)
            callTrain.append(ctypeTest)

        
        # Now for all the other combinations
        for ctypeTrain in callTypeUnique:
            if ctypeTrain == ctypeTest:
                continue
            ctypeIndBird1 = (callTypezPair == ctypeTrain) & (birdzPair == gp['bird1'])
            ctypeIndBird2 = (callTypezPair == ctypeTrain) & (birdzPair == gp['bird2'])
            countType1 = np.sum(ctypeIndBird1)
            countType2 = np.sum(ctypeIndBird2)
            if countType1 < MINCOUNT or countType2 < MINCOUNT:
                continue
            trainInd = np.asarray(callTypezPair == ctypeTrain).nonzero()[0]

            # Generate the index for training and testing that discriminatePlot needs
            allInd = np.hstack((trainInd, testInd))
            testIndTF = np.asarray([False] * len(trainInd) + [True] *len(testInd))

            for fl in featureList:
                        
                # Perform pair-wise discrimination across calls
                ldaY, qdaY, rfY, cvC, ldaP, qdaP, rfP, nC, ldaweights = discriminatePlot(fl['X'][allInd], birdzPair[allInd], cValBirdPair[allInd], 
                                                                                 titleStr='Voice %s (%s vs %s) Spectro' % (ctypeTrain, gp['bird1'], gp['bird2']), 
                                                                                 figdir = figdir, testInd = testIndTF)
                if ldaY == -1:
                    print ('Error: Failure in discriminate Plot')
                    continue
                
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
                callTest.append(ctypeTest)
                callTrain.append(ctypeTrain)
            
            
d = {'CallTrain': np.array(callTrain),
     'CallTest' : np.array(callTest),
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

# Save data immediately
resultsDataFrame.to_hdf(outputTable, 'resultsDataFrame', mode='w')
                

#%% Generate aggregate Data Frames.  

### Warning this code might need an update. Ignores pair-wise discrimination.
alpha = 0.05
pnull = 0.5   # Assuming two classes - p value for null hypothesis

# Find the index that are unique to each Type
resultsGroupedAll = resultsDataFrame.groupby(['CallTrain', 'CallTest', 'Features'])
resultsGroupedSex = resultsDataFrame.groupby(['CallTrain', 'CallTest', 'Features', 'SexPair'])

# # Make an Aggregate Pandas Data Frame by hand.  Pandas aggregate did not work well with mixed data.
numberOfRows = len(resultsGroupedAll.indices) + len(resultsGroupedSex.indices)
resultsAgg = pandas.DataFrame(index=np.arange(0, numberOfRows), 
            columns=('CallTrain', 'CallTest', 'Features', 'SexPair', 'NPairs',
                     'LDA', 'LDA_P', 'QDA', 'QDA_P', 
                     'RF', 'RF_P', 'Weights', 'Weights_SE') )

i = 0
for key in resultsGroupedAll.indices:
       
    # Extract CallTrain Type from key
    resultsAgg.iloc[i,0] = key[0]   # CallType
    
    # Extract CallTest Type from key
    resultsAgg.iloc[i,1] = key[1]   # Test Type
    
    # Extract Features from key
    resultsAgg.iloc[i,2] = key[2]
    
    # Sex Pair Code
    resultsAgg.iloc[i,3] = 'A'   # A for all: M, F, X and U

    # Number of pair-wise comparisons
    n = len(resultsGroupedAll.indices[key])
    resultsAgg.iloc[i,4] = n # NPairs
    
    # Loop through pairs to look at number that are significant
    ldanSig = 0
    ldaPavg = 0
    qdanSig = 0
    qdaPavg = 0
    rfnSig = 0
    rfPavg = 0
    
    for ind in resultsGroupedAll.indices[key]:
        ntest = resultsDataFrame.Count.iloc[ind]
        
        # Stats for LDA
        nLDAYes = resultsDataFrame.LDAYes.iloc[ind]
        ldaPavg += float(nLDAYes)/ntest
        
        ldaP = 0
        for k in range(nLDAYes, ntest+1):
            ldaP += binom.pmf(k, ntest, pnull)           
        if ldaP < alpha:
            ldanSig += 1
                
        # Stats for QDA
        nQDAYes = resultsDataFrame.QDAYes.iloc[ind]
        qdaPavg += float(nQDAYes)/ntest
        
        qdaP = 0
        for k in range(nQDAYes, ntest+1):
            qdaP += binom.pmf(k, ntest, pnull)           
        if qdaP < alpha:
            qdanSig += 1
            
        # Stats for RF
        nrfYes = resultsDataFrame.QDAYes.iloc[ind]
        rfPavg += float(nrfYes)/ntest
        
        rfP = 0
        for k in range(nrfYes, ntest+1):
            rfP += binom.pmf(k, ntest, pnull)           
        if rfP < alpha:
            rfnSig += 1
     
    # Average Performances
    ldaPavg /= n
    qdaPavg /= n
    rfPavg /= n
    
    # Number of significant pairs
    ldaP = 0
    for k in range(ldanSig, n+1):
        ldaP += binom.pmf(k, n, pnull)           
        
    qdaP = 0
    for k in range(qdanSig, n+1):
        qdaP += binom.pmf(k, n, pnull)           

    rfP = 0
    for k in range(rfnSig, n+1):
        rfP += binom.pmf(k, n, pnull) 
        
    # Stpre Data
    resultsAgg.iloc[i,5] = ldaPavg
    resultsAgg.iloc[i,6] = ldaP
    resultsAgg.iloc[i,7] = qdaPavg
    resultsAgg.iloc[i,8] = qdaP
    resultsAgg.iloc[i,9] = rfPavg
    resultsAgg.iloc[i,10] = rfP

                   
    # Average Weights and SE
    WeightsAll = np.vstack(resultsDataFrame.Weights.iloc[resultsGroupedAll.indices[key]])
    countAll = np.array(resultsDataFrame.Count.iloc[resultsGroupedAll.indices[key]], ndmin=2 )
    WeightsMean = np.dot(countAll, WeightsAll)/np.sum(countAll)
    
    WeightsSE = np.sqrt(np.dot(countAll, (WeightsAll-WeightsMean)**2)/(np.sum(countAll)*n))
    resultsAgg.iloc[i,11] = WeightsMean
    resultsAgg.iloc[i,12] = WeightsSE               
                   
    i += 1
    
for key in resultsGroupedSex.indices:
       
    # Extract CallTrain Type from key
    resultsAgg.iloc[i,0] = key[0]   # CallType
    
    # Extract CallTest type from key
    resultsAgg.iloc[i,1] = key[1]
    
    # Extract Features from key
    resultsAgg.iloc[i,2] = key[2]
    
    # Sex Pair Code
    resultsAgg.iloc[i,3] = key[3]   # A M, F, X or U
    
    # Number of pair-wise comparisons
    n = len(resultsGroupedSex.indices[key])
    resultsAgg.iloc[i,4] = n # NPairs
    
    # Loop through pairs to look at number that are significant
    ldanSig = 0
    ldaPavg = 0
    qdanSig = 0
    qdaPavg = 0
    rfnSig = 0
    rfPavg = 0
    
    for ind in resultsGroupedSex.indices[key]:
        ntest = resultsDataFrame.Count.iloc[ind]
        
        # Stats for LDA
        nLDAYes = resultsDataFrame.LDAYes.iloc[ind]
        ldaPavg += float(nLDAYes)/ntest
        
        ldaP = 0
        for k in range(nLDAYes, ntest+1):
            ldaP += binom.pmf(k, ntest, pnull)           
        if ldaP < alpha:
            ldanSig += 1
                
        # Stats for QDA
        nQDAYes = resultsDataFrame.QDAYes.iloc[ind]
        qdaPavg += float(nQDAYes)/ntest
        
        qdaP = 0
        for k in range(nQDAYes, ntest+1):
            qdaP += binom.pmf(k, ntest, pnull)           
        if qdaP < alpha:
            qdanSig += 1
            
        # Stats for RF
        nrfYes = resultsDataFrame.QDAYes.iloc[ind]
        rfPavg += float(nrfYes)/ntest
        
        rfP = 0
        for k in range(nrfYes, ntest+1):
            rfP += binom.pmf(k, ntest, pnull)           
        if rfP < alpha:
            rfnSig += 1
     
    # Average Performances
    ldaPavg /= n
    qdaPavg /= n
    rfPavg /= n
    
    # Number of significant pairs
    ldaP = 0
    for k in range(ldanSig, n+1):
        ldaP += binom.pmf(k, n, pnull)           
        
    qdaP = 0
    for k in range(qdanSig, n+1):
        qdaP += binom.pmf(k, n, pnull)           

    rfP = 0
    for k in range(rfnSig, n+1):
        rfP += binom.pmf(k, n, pnull) 
        
    # Stpre Data
    resultsAgg.iloc[i,5] = ldaPavg
    resultsAgg.iloc[i,6] = ldaP
    resultsAgg.iloc[i,7] = qdaPavg
    resultsAgg.iloc[i,8] = qdaP
    resultsAgg.iloc[i,9] = rfPavg
    resultsAgg.iloc[i,10] = rfP

                   
    # Average Weights and SE
    WeightsAll = np.vstack(resultsDataFrame.Weights.iloc[resultsGroupedSex.indices[key]])
    countAll = np.array(resultsDataFrame.Count.iloc[resultsGroupedSex.indices[key]], ndmin=2 )
    WeightsMean = np.dot(countAll, WeightsAll)/np.sum(countAll)
    
    WeightsSE = np.sqrt(np.dot(countAll, (WeightsAll-WeightsMean)**2)/(np.sum(countAll)*n))
    resultsAgg.iloc[i,11] = WeightsMean
    resultsAgg.iloc[i,12] = WeightsSE               
                   
    i += 1

# Print average performances of Classifiers
print('Average (per callpair - across all')
print('LDA:', resultsAgg.LDA.mean(), ' QDA:', resultsAgg.QDA.mean(), ' RF:', resultsAgg.RF.mean() )

# Save Data

resultsAgg.to_hdf(outputGroupedTable, 'resultsAgg', mode='w')

# Write the results to Excel without the Weights
resultsDataFrame.to_excel(fileExcelTable, columns = resultsDataFrame.columns[0:11])

#%% Read Data - one could start here.
resultsDataFrame = pandas.read_hdf(outputTable)
resultsAgg = pandas.read_hdf(outputGroupedTable)

#%% Make a probability matrix to summarize results
alpha = 0.05

# Select rows for all sex combinations and all Features and adult calls only
indMatAllAdult = [index for index, row in resultsAgg.iterrows() if ( (row.SexPair == "A") and (row.Features == 'Spectro') and ( (row.CallTrain != 'LT' ) and (row.CallTrain != 'Be')) ) ]
indMatDiagAdult = [index for index, row in resultsAgg.iterrows() if ( (row.SexPair == "A") and (row.Features == 'Spectro') and (row.CallTrain == row.CallTest) and ( (row.CallTrain != 'LT' ) and (row.CallTrain != 'Be')) )]

resultsAllAdult = resultsAgg.loc[indMatAllAdult]
resultsDiagAdult = resultsAgg.loc[indMatDiagAdult]

# Sort data by descending performance for LDA discrimination
resultsDiagAdult = resultsDiagAdult.sort_values(by='LDA', ascending = False) 
sortedCallTypesAdult = resultsDiagAdult.CallTest   # Sorted Call Types
ncalls = len(sortedCallTypesAdult)

# Build Matrix for LDA
LDAMat = np.zeros((ncalls, ncalls))
LDAPMat = np.zeros((ncalls, ncalls))
QDAMat = np.zeros((ncalls, ncalls))
QDAPMat = np.zeros((ncalls, ncalls))
RFMat = np.zeros((ncalls, ncalls))
RFPMat = np.zeros((ncalls, ncalls))

for i in range(ncalls):
    for j in range(ncalls):
        if i == j:
            LDAMat[i,j] = resultsDiagAdult.iloc[i].LDA
            QDAMat[i,j] = resultsDiagAdult.iloc[i].QDA
            RFMat[i,j] = resultsDiagAdult.iloc[i].RF
            if resultsDiagAdult.iloc[i].LDA_P < alpha :
                LDAPMat[i,j] = 1
            if resultsDiagAdult.iloc[i].QDA_P < alpha :
                QDAPMat[i,j] = 1
            if resultsDiagAdult.iloc[i].RF_P < alpha :
                RFPMat[i,j] = 1
        else:
            row = [row for index, row in resultsAllAdult.iterrows() if ((row.CallTrain == sortedCallTypesAdult.iloc[i]) and (row.CallTest == sortedCallTypesAdult.iloc[j]))]
            if len(row) == 1:
                LDAMat[i,j] = row[0].LDA
                QDAMat[i,j] = row[0].QDA
                RFMat[i,j] = row[0].RF
                if row[0].LDA_P < alpha :
                    LDAPMat[i,j] = 1
                if row[0].QDA_P < alpha :
                    QDAPMat[i,j] = 1
                if row[0].RF_P < alpha :
                    RFPMat[i,j] = 1
            else:
                print('Error missing entry for Training %s and Testing %s' % (sortedCallTypesAdult.iloc[j], sortedCallTypesAdult.iloc[i]))
 

# Color code for call type
callColor = {'Be': (0/255.0, 230/255.0, 255/255.0), 'LT': (0/255.0, 95/255.0, 255/255.0), 'Tu': (255/255.0, 200/255.0, 65/255.0), 'Th': (255/255.0, 150/255.0, 40/255.0), 
             'Di': (255/255.0, 105/255.0, 15/255.0), 'Ag': (255/255.0, 0/255.0, 0/255.0), 'Wh': (255/255.0, 180/255.0, 255/255.0), 'Ne': (255/255.0, 100/255.0, 255/255.0),
             'Te': (140/255.0, 100/255.0, 185/255.0), 'DC': (100/255.0, 50/255.0, 200/255.0), 'So': (0/255.0, 0/255.0, 0/255.0)}


plt.figure()
ax = plt.subplot(111)
plt.imshow(LDAMat, aspect='equal')
plt.xlabel('Testing')
plt.ylabel('Training')
ax.set_xticklabels(['']+sortedCallTypesAdult.tolist())
ax.set_yticklabels(['']+sortedCallTypesAdult.tolist())
for i in range(ncalls):
    for j in range(ncalls):
        if LDAPMat[i,j] :
            plt.text( j, i, '*', fontsize=15, color = 'r')
plt.clim(vmin=0.5, vmax = 1.0)
plt.title('LDA')
plt.colorbar()


plt.figure()

ax = plt.subplot(121)
plt.imshow(QDAMat, aspect='equal')
plt.xlabel('Testing')
plt.ylabel('Training')
ax.set_xticklabels(['']+sortedCallTypesAdult.tolist())
ax.set_yticklabels(['']+sortedCallTypesAdult.tolist())
for i in range(ncalls):
    for j in range(ncalls):
        if QDAPMat[i,j] :
            plt.text( j, i, '*', fontsize=15, color = 'r')
plt.clim(vmin=0.5, vmax = 1.0)
plt.title('QDA')
plt.colorbar()

ax = plt.subplot(122)
plt.imshow(RFMat, aspect='equal')
plt.xlabel('Testing')
plt.ylabel('Training')
ax.set_xticklabels(['']+sortedCallTypesAdult.tolist())
ax.set_yticklabels(['']+sortedCallTypesAdult.tolist())
for i in range(ncalls):
    for j in range(ncalls):
        if RFPMat[i,j] :
            plt.text( j, i, '*', fontsize=15, color = 'r')
plt.clim(vmin=0.5, vmax = 1.0)
plt.title('RF')

plt.colorbar()


plt.show()

