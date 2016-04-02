from lasp.sound import BioSound 
import os
import pandas
import numpy as np


# Read the matlab file that has all the cut sounds
os.chdir('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/h5files')
store = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/vocSelTable.h5'

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
        Bird = np.array2string(myBioSound.emitter).translate(None, "'")
        # Clean up the data
        if Bird is 'HpiHpi4748':
            Bird = 'HPiHPi4748'
        
        callType = np.array2string(myBioSound.type).translate(None, "'")
        # Clean up the data
        if callType == 'C-' or callType == 'WC':  # C- correspondsto unknown-11 and WC are copulation whines
            continue
        if callType == '-A':
            callType = 'Ag'
        
        if myBioSound.fund is not None:
            fund = np.float(myBioSound.fund)
            cvfund = np.float(myBioSound.cvfund)
        else:
            fund = None 
            cvfund = None   

        vocSelData.append({"Bird": Bird, "calltype": callType, "fund": fund, 
                     "cvfund": cvfund, "F1": meanF1, "F2": meanF2, "F3":meanF3,
                     "sal": np.float(myBioSound.sal), "rms": np.float(myBioSound.rms), 
                     "meanS": np.float(myBioSound.meanspect), "stdS": np.float(myBioSound.stdspect), 
                     "q1": np.float(myBioSound.q1), "q2": np.float(myBioSound.q2), "q3": np.float(myBioSound.q3),
                     "timestd": np.float(myBioSound.stdtime)})

# Make two data frames one with all the data and one grouped by Bird and calltype                    
vocSelTable = pandas.DataFrame(vocSelData)
vocSelTable.to_hdf(store, 'callTable', mode = 'w')
