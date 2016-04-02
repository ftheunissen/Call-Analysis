from lasp.sound import BioSound 
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt


store = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/vocSelTable.h5'
vocSelTable = pandas.read_hdf(store)

# Make two data frames one with all the data and one grouped by Bird and calltype                    
vocSelTableGrouped = vocSelTable.groupby(['Bird','calltype'])
vocSelTableGroupedAgg = vocSelTableGrouped.aggregate('mean').reset_index()


# Plot some nice graphs
# Color code for call type
callColor = {'Be': (0/255.0, 230/255.0, 255/255.0), 'LT': (0/255.0, 95/255.0, 255/255.0), 'Tu': (255/255.0, 200/255.0, 65/255.0), 'Th': (255/255.0, 150/255.0, 40/255.0), 
             'Di': (255/255.0, 105/255.0, 15/255.0), 'Ag': (255/255.0, 0/255.0, 0/255.0), 'Wh': (255/255.0, 180/255.0, 255/255.0), 'Ne': (255/255.0, 100/255.0, 255/255.0),
             'Te': (140/255.0, 100/255.0, 185/255.0), 'DC': (100/255.0, 50/255.0, 200/255.0), 'So': (255/255.0, 255/255.0, 255/255.0)}

callTypes = np.unique(vocSelTableGroupedAgg['calltype'])

cVal = []
for cType in vocSelTableGroupedAgg['calltype']:
    cVal.append(callColor[cType])
    
# Color code for bird ID
birdNames = np.unique(vocSelTableGroupedAgg['Bird'])

birdColor = {}
for birdId in birdNames:
    birdColor[birdId] = np.random.rand(3)
    
cValBirdAll = []
for birdId in vocSelTable['Bird']:
    cValBirdAll.append(birdColor[birdId])
    
cValBird = []
for birdId in vocSelTableGroupedAgg['Bird']:
    cValBird.append(birdColor[birdId])

# Display formants
plt.figure()
ax = plt.subplot(131)
plt.scatter(vocSelTableGroupedAgg['F1'], vocSelTableGroupedAgg['F2'], c=cVal, s=40)
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('F1 (Hz)')
plt.ylabel('F2 (Hz)')
plt.xlim((1000, 6000))
plt.ylim((1000, 6000))
ax = plt.subplot(132)
plt.scatter(vocSelTableGroupedAgg['F1'], vocSelTableGroupedAgg['F3'], c=cVal, s=40)
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('F1 (Hz)')
plt.ylabel('F3 (Hz)')
plt.xlim((1000, 6000))
plt.ylim((1000, 6000))
ax = plt.subplot(133)
plt.scatter(vocSelTableGroupedAgg['F2'], vocSelTableGroupedAgg['F3'], c=cVal, s=40)
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('F2 (Hz)')
plt.ylabel('F3 (Hz)')
plt.xlim((1000, 6000))
plt.ylim((1000, 6000))
plt.show()

# Display Saliency vs Formants
plt.figure()
plt.subplot(131)
plt.scatter(vocSelTableGroupedAgg['sal'], vocSelTableGroupedAgg['F1'], c=cVal, s=40)
plt.xlabel('sal')
plt.ylabel('F1 (Hz)')
plt.subplot(132)
plt.scatter(vocSelTableGroupedAgg['sal'], vocSelTableGroupedAgg['F2'], c=cVal, s=40)
plt.xlabel('sal')
plt.ylabel('F2 (Hz)')
plt.subplot(133)
plt.scatter(vocSelTableGroupedAgg['sal'], vocSelTableGroupedAgg['F3'], c=cVal, s=40)
plt.xlabel('sal')
plt.ylabel('F3 (Hz)')
plt.show()


plt.figure()
plt.subplot(131)
plt.scatter(vocSelTableGroupedAgg['F1'], vocSelTableGroupedAgg['F2']-(1.0*vocSelTableGroupedAgg['F1']), c=cVal, s=40)
plt.xlabel('F1 (Hz)')
plt.ylabel('F2 - F1 (Hz)')
plt.subplot(132)
plt.scatter(vocSelTableGroupedAgg['F1'], vocSelTableGroupedAgg['F3']-(2.0*vocSelTableGroupedAgg['F1']), c=cVal, s=40)
plt.xlabel('F1 (Hz)')
plt.ylabel('F3 - 2*F1 (Hz)')
plt.subplot(133)
plt.scatter(vocSelTableGroupedAgg['F2'], vocSelTableGroupedAgg['F3']-(1.0*vocSelTableGroupedAgg['F2']), c=cVal, s=40)
plt.xlabel('F2 (Hz)')
plt.ylabel('F3 - F2 (Hz)')
plt.show()

plt.figure()
plt.subplot(131)
plt.scatter(vocSelTableGroupedAgg['q1'], vocSelTableGroupedAgg['q2'], c=cVal, s=40)
plt.xlabel('Q1 (Hz)')
plt.ylabel('Q2 (Hz)')
plt.subplot(132)
plt.scatter(vocSelTableGroupedAgg['q1'], vocSelTableGroupedAgg['q3'], c=cVal, s=40)
plt.xlabel('Q1 (Hz)')
plt.ylabel('Q3 (Hz)')
plt.subplot(133)
plt.scatter(vocSelTableGroupedAgg['q2'], vocSelTableGroupedAgg['q3'], c=cVal, s=40)
plt.xlabel('Q2 (Hz)')
plt.ylabel('Q3 (Hz)')
plt.show()

