from soundsig.sound import BioSound 
import os
import pandas
import numpy as np



# Specify the directory that has the h5 files for all cut vocalizations
os.chdir('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/h5files')
store = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/vocParamTable.h5'

nfiles = len([fname for fname in os.listdir('.') if fname.endswith('h5')])

# Read the bird information file
birdInfo = np.genfromtxt('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/Birds_List_Acoustic.txt', dtype=None);
birdInfoNames = [ rowdata[0] for rowdata in birdInfo]

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
            maxfund = np.float(myBioSound.maxfund)
            minfund = np.float(myBioSound.minfund)
        else:
            fund = None 
            cvfund = None   
            
        # Find bird name in birdInfo.
        birdInfoLocal = birdInfo[np.in1d(birdInfoNames, Bird)]
        if len(birdInfoLocal) != 0:
            sex = birdInfoLocal[0][1]
            agestr = birdInfoLocal[0][2]
        else:
            sex = 'U'
            agestr = 'U'

        vocSelData.append({"Bird": Bird, "Sex": sex, "Age": agestr,
                     "calltype": callType, "fund": fund, 
                     "cvfund": cvfund, "maxfund": maxfund, "minfund": minfund,
                     "F1": meanF1, "F2": meanF2, "F3":meanF3,
                     "sal": np.float(myBioSound.sal), "rms": np.float(myBioSound.rms), 
                     "maxAmp": np.float(myBioSound.maxAmp),
                     "meanS": np.float(myBioSound.meanspect), "stdS": np.float(myBioSound.stdspect),
                     "skewS": np.float(myBioSound.skewspect), "kurtS": np.float(myBioSound.kurtosisspect), 
                     "entS": np.float(myBioSound.entropyspect),
                     "q1": np.float(myBioSound.q1), "q2": np.float(myBioSound.q2), "q3": np.float(myBioSound.q3),                  
                     "meanT": np.float(myBioSound.meantime), "stdT": np.float(myBioSound.stdtime),
                     "skewT": np.float(myBioSound.skewtime), "kurtT": np.float(myBioSound.kurtosistime),
                     "entT": np.float(myBioSound.entropytime)
})

# Make a panda data frame with all the data                    
vocSelTable = pandas.DataFrame(vocSelData)
vocSelTable.to_hdf(store, 'callTable', mode = 'w')
