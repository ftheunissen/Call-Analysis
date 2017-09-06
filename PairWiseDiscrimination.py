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

#%% Stores 
inputTable = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/vocParamTable.h5'
outputTable = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerDiscrimResults.h5'
outputGroupedTable = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerDiscrimGroupedResults.h5'
fileExcelTable = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerDiscrimResults.xls'
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
ldaYes = []
qdaYes = []
rfYes = [] 
cvCount = []
ldaProb = []
qdaProb = []
rfProb = []
nClasses = []
Type = []
Features = []
Weights = []
SexPair = []
BirdPair = []

# Loop through call types:
for ctype in callTypes:
    goodInd = callTypez == ctype
    goodInd = goodInd.nonzero()
    nsamples = np.size(goodInd)
    birds = np.unique(birdz[goodInd])
    nbirds = birds.size
    print 'Performing Voice Discrimination for %s with %d samples from %d birds' % (ctype, nsamples, nbirds)
    if (nbirds < 2) :
        print 'Warning: Insuficient number of birds to perform discrimination'
    else:
        for i1 in range(nbirds):
            bird1 = birds[i1]
            for i2 in range(i1+1, nbirds):
                bird2 = birds[i2]
                birdPairInd = (callTypez == ctype) & ((birdz == bird1) | (birdz == bird2))
                                
                # Perform pair-wise discrimination
                ldaY, qdaY, rfY, cvC, ldaP, qdaP, rfP, nC, ldaweights = discriminatePlot(Xallz[birdPairInd], birdz[birdPairInd], cValBirdAll[birdPairInd], titleStr='Caller %s (%s vs %s) 18 AF' % (ctype, bird1, bird2), figdir = figdir, Xcolname = XallNames)
        
                if ldaY == -1:
                    continue
                
                # Sex labeling
                sexPairs = np.unique(birdSex[birdPairInd])
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
            
                # Store data for complete
                ldaYes.append(ldaY)
                ldaProb.append(ldaP)
                qdaYes.append(qdaY)
                qdaProb.append(qdaP)
                rfYes.append(rfY) 
                rfProb.append(rfP)
                cvCount.append(cvC)
                nClasses.append(nC)
                Features.append('18 AF')
                Weights.append(abs(ldaweights).mean(axis=0))
                SexPair.append(sexLabel)
                BirdPair.append((bird1, bird2))
                Type.append('Caller %s' % ctype)
                
                # Store data for fundamental only
                ldaY, qdaY, rfY, cvC, ldaP, qdaP, rfP, nC, ldaweights = discriminatePlot(Xfundz[birdPairInd], birdz[birdPairInd], cValBirdAll[birdPairInd], titleStr='Caller %s (%s vs %s) Fund' % (ctype, bird1, bird2), figdir = figdir, Xcolname = XfundNames)

                ldaYes.append(ldaY)
                ldaProb.append(ldaP)
                qdaYes.append(qdaY)
                qdaProb.append(qdaP)
                rfYes.append(rfY) 
                rfProb.append(rfP)
                cvCount.append(cvC)
                nClasses.append(nC)
                Features.append('Fund AF')
                Weights.append(abs(ldaweights).mean(axis=0))
                SexPair.append(sexLabel)
                BirdPair.append((bird1, bird2))
                Type.append('Caller %s' % ctype)
                
                # Store data for spectral only
                ldaY, qdaY, rfY, cvC, ldaP, qdaP, rfP, nC, ldaweights = discriminatePlot(Xspectz[birdPairInd], birdz[birdPairInd], cValBirdAll[birdPairInd], titleStr='Caller %s (%s vs %s) Spect' % (ctype, bird1, bird2), figdir = figdir, Xcolname = XspectNames)

                ldaYes.append(ldaY)
                ldaProb.append(ldaP)
                qdaYes.append(qdaY)
                qdaProb.append(qdaP)
                rfYes.append(rfY) 
                rfProb.append(rfP)
                cvCount.append(cvC)
                nClasses.append(nC)
                Features.append('Spect AF')
                Weights.append(abs(ldaweights).mean(axis=0))
                SexPair.append(sexLabel)
                BirdPair.append((bird1, bird2))
                Type.append('Caller %s' % ctype)
                
                # Store data for spectral only
                ldaY, qdaY, rfY, cvC, ldaP, qdaP, rfP, nC, ldaweights = discriminatePlot(Xtempz[birdPairInd], birdz[birdPairInd], cValBirdAll[birdPairInd], titleStr='Caller %s (%s vs %s) Temp' % (ctype, bird1, bird2), figdir = figdir, Xcolname = XtempNames)

                ldaYes.append(ldaY)
                ldaProb.append(ldaP)
                qdaYes.append(qdaY)
                qdaProb.append(qdaP)
                rfYes.append(rfY) 
                rfProb.append(rfP)
                cvCount.append(cvC)
                nClasses.append(nC)
                Features.append('Temp AF')
                Weights.append(abs(ldaweights).mean(axis=0))
                SexPair.append(sexLabel)
                BirdPair.append((bird1, bird2))
                Type.append('Caller %s' % ctype)
                
                        
        
d = {'Type': np.array(Type),
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

# Find the index that are unique to each Type
resultsGroupedAll = resultsDataFrame.groupby(['Type', 'Features'])
resultsGroupedSex = resultsDataFrame.groupby(['Type', 'Features', 'SexPair'])

# # Make an Aggregate Pandas Data Frame by hand.  Pandas aggregate did not work well with mixed data.
numberOfRows = len(resultsGroupedAll.indices) + len(resultsGroupedSex.indices)
resultsAgg = pandas.DataFrame(index=np.arange(0, numberOfRows), 
            columns=('CallType', 'Features', 'SexPair', 'NPairs', 'TestCount',
                     'LDA', 'LDA_YES', 'LDA_P', 'QDA', 'QDA_YES', 'QDA_P', 
                     'RF', 'RF_YES', 'RF_P', 'Weights', 'Weights_SE') )

i = 0
for key in resultsGroupedAll.indices:
       
    # Extract Call Type from key
    resultsAgg.iloc[i,0] = key[0][-2:]   # CallType
    
    # Extract Features from key
    resultsAgg.iloc[i,1] = key[1]
    
    # Sex Pair Code
    resultsAgg.iloc[i,2] = 'A'   # A for all: M, F, X and U

    # Number of pair-wise comparisons
    n = len(resultsGroupedAll.indices[key])
    resultsAgg.iloc[i,3] = n # NPairs
    
    # Number of trials used in cross-validation
    resultsAgg.iloc[i,4] = resultsDataFrame.Count.iloc[resultsGroupedAll.indices[key]].sum()
    
    # LDA_YES and LDA_P
    resultsAgg.iloc[i,6] = resultsDataFrame.LDAYes.iloc[resultsGroupedAll.indices[key]].sum() 
    resultsAgg.iloc[i,5] =  100.0*resultsAgg.iloc[i,6]/resultsAgg.iloc[i,4]
    ldaP = 0
    p = 0.5   # Assuming 2 classes here
    for k in range(resultsAgg.iloc[i,6], resultsAgg.iloc[i,4]+1):
        ldaP += binom.pmf(k, resultsAgg.iloc[i,4], p)
    resultsAgg.iloc[i,7] = ldaP
    
    # QDA_YES and QDA_P
    resultsAgg.iloc[i,9] = resultsDataFrame.QDAYes.iloc[resultsGroupedAll.indices[key]].sum() 
    resultsAgg.iloc[i,8] =  100.0*resultsAgg.iloc[i,9]/resultsAgg.iloc[i,4]
    qdaP = 0
    p = 0.5   # Assuming 2 classes here
    for k in range(resultsAgg.iloc[i,9], resultsAgg.iloc[i,4]+1):
        qdaP += binom.pmf(k, resultsAgg.iloc[i,4], p)
    resultsAgg.iloc[i,10] = qdaP
       
    # RF and RF_SE
    resultsAgg.iloc[i,12] = resultsDataFrame.RFYes.iloc[resultsGroupedAll.indices[key]].sum()
    resultsAgg.iloc[i,11] =  100.0*resultsAgg.iloc[i,12]/resultsAgg.iloc[i,4]
    rfP = 0
    p = 0.5   # Assuming 2 classes here
    for k in range(resultsAgg.iloc[i,12], resultsAgg.iloc[i,4]+1):
        rfP += binom.pmf(k, resultsAgg.iloc[i,4], p)
    resultsAgg.iloc[i,13] = rfP
                   
    # Average Weights and SE
    WeightsAll = np.vstack(resultsDataFrame.Weights.iloc[resultsGroupedAll.indices[key]])
    countAll = np.array(resultsDataFrame.Count.iloc[resultsGroupedAll.indices[key]], ndmin=2 )
    WeightsMean = np.dot(countAll, WeightsAll)/np.sum(countAll)
    
    WeightsSE = np.sqrt(np.dot(countAll, (WeightsAll-WeightsMean)**2)/(np.sum(countAll)*n))
    resultsAgg.iloc[i,14] = WeightsMean
    resultsAgg.iloc[i,15] = WeightsSE               
                   
    i += 1
    
for key in resultsGroupedSex.indices:
       
    # Extract Call Type from key
    resultsAgg.iloc[i,0] = key[0][-2:]   # CallType
    
    # Extract Features from key
    resultsAgg.iloc[i,1] = key[1]
    
    # Sex Pair Code
    resultsAgg.iloc[i,2] = key[2]   # A M, F, X or U

    # Number of pair-wise comparisons
    n = len(resultsGroupedSex.indices[key])
    resultsAgg.iloc[i,3] = n # NPairs
    
    # Number of trials used in cross-validation
    resultsAgg.iloc[i,4] = resultsDataFrame.Count.iloc[resultsGroupedSex.indices[key]].sum()
    
    # LDA_YES and LDA_P
    resultsAgg.iloc[i,6] = resultsDataFrame.LDAYes.iloc[resultsGroupedSex.indices[key]].sum() 
    resultsAgg.iloc[i,5] =  100.0*resultsAgg.iloc[i,6]/resultsAgg.iloc[i,4]
    ldaP = 0
    p = 0.5   # Assuming 2 classes here
    for k in range(resultsAgg.iloc[i,6], resultsAgg.iloc[i,4]+1):
        ldaP += binom.pmf(k, resultsAgg.iloc[i,4], p)
    resultsAgg.iloc[i,7] = ldaP
    
    # QDA_YES and QDA_P
    resultsAgg.iloc[i,9] = resultsDataFrame.QDAYes.iloc[resultsGroupedSex.indices[key]].sum() 
    resultsAgg.iloc[i,8] =  100.0*resultsAgg.iloc[i,9]/resultsAgg.iloc[i,4]
    qdaP = 0
    p = 0.5   # Assuming 2 classes here
    for k in range(resultsAgg.iloc[i,9], resultsAgg.iloc[i,4]+1):
        qdaP += binom.pmf(k, resultsAgg.iloc[i,4], p)
    resultsAgg.iloc[i,10] = qdaP
       
    # RF and RF_SE
    resultsAgg.iloc[i,12] = resultsDataFrame.RFYes.iloc[resultsGroupedSex.indices[key]].sum()
    resultsAgg.iloc[i,11] =  100.0*resultsAgg.iloc[i,12]/resultsAgg.iloc[i,4]
    rfP = 0
    p = 0.5   # Assuming 2 classes here
    for k in range(resultsAgg.iloc[i,12], resultsAgg.iloc[i,4]+1):
        rfP += binom.pmf(k, resultsAgg.iloc[i,4], p)
    resultsAgg.iloc[i,13] = rfP
                   
    # Average Weights and SE
    WeightsAll = np.vstack(resultsDataFrame.Weights.iloc[resultsGroupedSex.indices[key]])
    countAll = np.array(resultsDataFrame.Count.iloc[resultsGroupedSex.indices[key]], ndmin=2 )
    WeightsMean = np.dot(countAll, WeightsAll)/np.sum(countAll)
    
    WeightsSE = np.sqrt(np.dot(countAll, (WeightsAll-WeightsMean)**2)/(np.sum(countAll)*n))
    resultsAgg.iloc[i,14] = WeightsMean
    resultsAgg.iloc[i,15] = WeightsSE                 
                   
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
resultsDataFrame.to_excel(fileExcelTable, columns = resultsDataFrame.columns[0:11])

#%% Make a plot
# A for both male and females.
indCallerA = [index for index, row in resultsAgg.iterrows() if ( (row.SexPair == "A") and (row.Features == '18 AF'))]
resultsperCallTypeA = resultsAgg.loc[indCallerA]
# Sort data by descending performance for LDA discrimination
resultsperCallTypeA = resultsperCallTypeA.sort_values(by='LDA', ascending = False) 
sortedCallTypes = resultsperCallTypeA.CallType   # Sorted Call Types

# M for Males
indCallerM = []
xvalsM = []
xvals = -1
oldlen = 0
for lbl in sortedCallTypes:
    xvals += 1
    indCallerM += [index for index, row in resultsAgg.iterrows() if ( (row.SexPair == 'M') and (row.CallType == lbl) and (row.Features == '18 AF')) ]
    newlen = len(indCallerM)
    if (newlen != oldlen):   # This code is for the x axis for the plot to skip if there is no data.
        xvalsM += [xvals]
        oldlen = newlen
resultsperCallTypeM = resultsAgg.loc[indCallerM]

# F for Females
indCallerF = []
xvalsF = []
xvals = -1
oldlen = 0
for lbl in sortedCallTypes:
    xvals += 1
    indCallerF += [index for index, row in resultsAgg.iterrows() if ( (row.SexPair == 'F') and (row.CallType == lbl) and (row.Features == '18 AF') ) ]
    newlen = len(indCallerF)
    if (newlen != oldlen):
        xvalsF += [xvals]
        oldlen = newlen
resultsperCallTypeF = resultsAgg.loc[indCallerF]

# X for Cross
indCallerX = []
xvalsX = []
xvals = -1
oldlen = 0
for lbl in sortedCallTypes:
    xvals += 1
    indCallerX += [index for index, row in resultsAgg.iterrows() if ( (row.SexPair == 'X') and (row.CallType == lbl) and (row.Features == '18 AF')) ]
    newlen = len(indCallerX)
    if (newlen != oldlen):
        xvalsX += [xvals]
        oldlen = newlen
resultsperCallTypeX = resultsAgg.loc[indCallerX]

# U for Unknown
indCallerU = []
xvalsU = []
xvals = -1
oldlen = 0
for lbl in sortedCallTypes:
    xvals += 1
    indCallerU += [index for index, row in resultsAgg.iterrows() if ( (row.SexPair == 'U') and (row.CallType == lbl) and (row.Features == '18 AF')) ]
    newlen = len(indCallerU)
    if (newlen != oldlen):
        xvalsU += [xvals]
        oldlen = newlen
resultsperCallTypeU = resultsAgg.loc[indCallerU]

plt.figure()
xvals = np.arange(len(indCallerA))
width = 0.75          # the width of the bars
# b1=plt.bar(xvals*5, resultsperCallTypeA.LDA, width, color='k', yerr=np.array(resultsperCallTypeA.LDA_SE))

b1=plt.bar(xvals*5, resultsperCallTypeA.LDA, width, color='k', yerr=10.0*np.ones(len(xvals)))

b2=plt.bar(np.array(xvalsM)*5 + width, resultsperCallTypeM.LDA, width, color='b', yerr=10.0*np.ones(len(xvalsM)))                      
b3=plt.bar(np.array(xvalsF)*5 + 2*width, resultsperCallTypeF.LDA, width, color='r', yerr=10.0*np.ones(len(xvalsF))) 
b4=plt.bar(np.array(xvalsX)*5 + 3*width, resultsperCallTypeX.LDA, width, color='g', yerr=10.0*np.ones(len(xvalsX))) 
b5=plt.bar(np.array(xvalsU)*5 + 4*width, resultsperCallTypeU.LDA, width, color='y', yerr=10.0*np.ones(len(xvalsU)))                      

plt.xticks(5*xvals + width*5/2., sortedCallTypes)
plt.legend((b1[0], b2[0], b3[0], b4[0], b5[0]), ('All', 'M', 'F', 'X', 'U'))
plt.title('Classification for Caller ID')
plt.ylabel('% ')
plt.xlabel('Call Type')
myAxis = plt.axis()
myAxis = (myAxis[0], myAxis[1], 50.0, myAxis[3])
plt.axis(myAxis)

plt.show()
