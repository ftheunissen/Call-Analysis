from lasp.sound import BioSound 
import os
import pandas
import numpy as np
from lasp.discriminate import discriminatePlot


# Read the matlab file that has all the cut sounds
os.chdir('/Users/frederictheunissen/Documents/Data/Babies/Banque Pleurs Francais originaux full /h5files')
store = '/Users/frederictheunissen/Documents/Data/Babies/Banque Pleurs Francais originaux full /vocSelTable.h5'

nfiles = len([fname for fname in os.listdir('.') if fname.endswith('h5')])

# Allocate space for fundamental, cvfund, formants, saliency, rms, spectral means, std, q1, q2, q3, time std
# Trying a list of dictionaries
vocSelData = []

for fname in os.listdir('.'):
    if fname.endswith('.h5'):
        myBioSound = BioSound()
        myBioSound.readh5(fname)
        meanF1 = np.mean(myBioSound.F1[~np.isnan(myBioSound.F1)])
        meanF2 = np.mean(myBioSound.F2[~np.isnan(myBioSound.F2)])
        meanF3 = np.mean(myBioSound.F3[~np.isnan(myBioSound.F3)])
        Baby = np.array2string(myBioSound.emitter).translate(None, "'")
        
        callType = np.array2string(myBioSound.type).translate(None, "'")
        
        if myBioSound.fund is not None:
            fund = np.float(myBioSound.fund)
            cvfund = np.float(myBioSound.cvfund)
        else:
            fund = None 
            cvfund = None   

        vocSelData.append({"Baby": Baby, "calltype": callType, "fund": fund, 
                     "cvfund": cvfund, "F1": meanF1, "F2": meanF2, "F3":meanF3,
                     "sal": np.float(myBioSound.sal), "rms": np.float(myBioSound.rms), 
                     "meanS": np.float(myBioSound.meanspect), "stdS": np.float(myBioSound.stdspect), 
                     "q1": np.float(myBioSound.q1), "q2": np.float(myBioSound.q2), "q3": np.float(myBioSound.q3),
                     "timestd": np.float(myBioSound.stdtime)})

# Make two data frames one with all the data and one grouped by Baby and calltype                    
vocSelTable = pandas.DataFrame(vocSelData)
vocSelTable.to_hdf(store, 'babyTable', mode = 'w')

vocSelTableGrouped = vocSelTable.groupby(['Baby','calltype'])
vocSelTableGroupedAgg = vocSelTableGrouped.aggregate('mean').reset_index()


# Color code for call type
callColor = {'F': (200/255.0, 0/255.0, 100/255.0), 'M': (0/255.0, 0/255.0, 255/255.0)}

callTypes = np.unique(vocSelTableGroupedAgg['calltype'])1

cVal = []
for cType in vocSelTableGroupedAgg['calltype']:
    cVal.append(callColor[cType])
    
 # Perform some classifications - with and without song - these are averaged per bird.
X = np.array([vocSelTableGroupedAgg['F1'], vocSelTableGroupedAgg['F2'], vocSelTableGroupedAgg['F3'], vocSelTableGroupedAgg['sal'], vocSelTableGroupedAgg['fund'], vocSelTableGroupedAgg['cvfund']])
y = np.array(vocSelTableGroupedAgg['calltype'])
cVal = np.asarray(cVal)   

# Spectral + Fund Model
ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses = discriminatePlot(X.transpose(), y, cVal, titleStr='M vs F Formants')

 # Perform some classifications - with and without song - these are averaged per bird.
X = np.array([vocSelTableGroupedAgg['q1'], vocSelTableGroupedAgg['q2'], vocSelTableGroupedAgg['q3'], vocSelTableGroupedAgg['sal'], vocSelTableGroupedAgg['fund'], vocSelTableGroupedAgg['cvfund']])
y = np.array(vocSelTableGroupedAgg['calltype'])
cVal = np.asarray(cVal)   

# Spectral + Fund Model
ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses = discriminatePlot(X.transpose(), y, cVal, titleStr='M vs F Quartiles')
