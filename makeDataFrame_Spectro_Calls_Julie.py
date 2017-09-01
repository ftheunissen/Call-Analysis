# Dependencies
from __future__ import print_function

import os
import pandas
import pickle
import numpy as np
from sklearn.decomposition import PCA

# Local dependencies
from soundsig.sound import BioSound 

# Specify the directory that has the h5 files for all cut vocalizations
os.chdir('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/h5files')
store = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/vocSpectroTable.h5'
pcInfo = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/vocSpectroPC.pkl'

nfiles = len([fname for fname in os.listdir('.') if fname.endswith('h5')])

# Read the bird information file
birdInfo = np.genfromtxt('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/Birds_List_Acoustic.txt', dtype=None);
birdInfoNames = [ rowdata[0] for rowdata in birdInfo]

# Allocate space for fundamental, cvfund, formants, saliency, rms, spectral means, std, q1, q2, q3, time std
# Trying a list of dictionaries
vocSelData = []

# Set true if you want to normalize all spectrograms.
normFlg = True
nPCs = 50

# Read firt one to allocate space for np array
count = 0
for fname in os.listdir('.'):
    if fname.endswith('.h5'):
        if count == 0:
            myBioSound = BioSound()
            myBioSound.readh5(fname)
        
            X = np.ravel(myBioSound.spectro)
        
        count += 1
        
X = np.zeros((count, X.shape[0]))
print('Allocate space for all spectrograms')       
    
        
# Preprocess spectrogram and perform dimensionality reduction
count = 0
for fname in os.listdir('.'):
    if fname.endswith('.h5'):
        myBioSound = BioSound()
        myBioSound.readh5(fname)

        # Massage spectrogram as in matlab code (DFA_Calls_Julie)
        if normFlg:  # Normalize by peak
            myBioSound.spectro -= myBioSound.spectro.max()

        # Set a 100 dB range threshold
        maxAmp = myBioSound.spectro.max();
        minAmp = maxAmp - 100;
        myBioSound.spectro[myBioSound.spectro < minAmp] = minAmp;

        X[count,:] = np.ravel(myBioSound.spectro)
        
        count +=1
            
print('Read %d files and spectrograms' % count)
print('Performing PCA')

pca = PCA(n_components=nPCs)
Xr = pca.fit_transform(X)  
 
# Write PCA information in pkl                         
pcInfoFile = open(pcInfo, 'wb')

pickle.dump(pca.components_, pcInfoFile)

print('PCA Done: Wrote PC\'s to pickle file %s' % pcInfoFile)
print ('Variance explained is %.2f%%' % (sum(pca.explained_variance_ratio_)*100.0))


        
count = 0
for fname in os.listdir('.'):
    if fname.endswith('.h5'):
        myBioSound = BioSound()
        myBioSound.readh5(fname)

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
          
            
        # Find bird name in birdInfo.
        birdInfoLocal = birdInfo[np.in1d(birdInfoNames, Bird)]
        if len(birdInfoLocal) != 0:
            sex = birdInfoLocal[0][1]
            agestr = birdInfoLocal[0][2]
        else:
            sex = 'U'
            agestr = 'U'
 
        
        vocSelData.append({"Bird": Bird, "Sex": sex, "Age": agestr,
                      "Calltype": callType, "Spectro": np.float(Xr[count])}) 
        count +=1      

# Make a panda data frame with all the data  
print('Reprocessed %d files to make Panda Data Frame'%count)                    
vocSelTable = pandas.DataFrame(vocSelData)
vocSelTable.to_hdf(store, 'callTable', mode = 'w')
print('Done: Wrote pandas data frame to h5 file %s' % store)
