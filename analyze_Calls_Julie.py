# Import math, plotting and sound libraries
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lasp.sound import BioSound 
import sys
import os

# Read the matlab file that has all the cut sounds
os.chdir('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/')
vocCutsContent = h5py.File('vocCuts.mat')

# This will be the output directory
if not os.path.exists('h5files'):
    os.makedirs('h5files')
os.chdir('h5files')

# load /auto/fdata/fet/julie/FullVocalizationBank/vocCuts.mat

# set the order in which the call types should be displayed in confusion
# matrices
name_grp_plot = ['Be', 'LT', 'Tu', 'Th', 'Di', 'Ag', 'Wh', 'Ne', 'Te', 'DC', 'So'];

# Initialize some of the values
soundCutsTot = np.array(vocCutsContent["soundCutsTot"])
spectroCutsTot = np.array(vocCutsContent["spectroCutsTot"])
samprate = np.array(vocCutsContent["samprate"]).squeeze()
vocTypeCuts = np.array(vocCutsContent['vocTypeCuts']).squeeze()
vocTypeCutInts = np.array([np.array(vocCutsContent[r]) for r in vocTypeCuts]).squeeze()
vocTypeCutsStrs = np.array([''.join(x) for x in vocTypeCutInts.view('c')])
birdNameCuts = np.array(vocCutsContent['birdNameCuts']).squeeze()
birdNameCutInts = np.array([np.array(vocCutsContent[r]) for r in birdNameCuts]).squeeze()
birdNameCutsStrs = np.array([''.join(x) for x in birdNameCutInts.view('c')])

fo = np.array(vocCutsContent["fo"]).squeeze()
to = np.array(vocCutsContent["to"]).squeeze()

nsounds = np.shape(soundCutsTot)[1]      # Number of calls in the library
plotme = 0

nf = max(np.shape(fo))   # Number of frequency slices in our spectrogram
nt = max(np.shape(to));      # Number of time slices in our spectrogram

# Loop through all sounds to extract temporal and spectral parameters


for inds in range(1407, nsounds):    
    #if vocTypeCutsStrs[inds] != 'DC':   #Distance call inds==7 is used as an example
    #    continue;
    
    print 'Processing sound %d/%d\n' % (inds, nsounds)
    sys.stdout.flush()
    
    soundIn  = soundCutsTot[:,inds]
    rms = np.std(soundIn[soundIn!=0]);
    soundSpect = np.reshape(spectroCutsTot[:,inds], (nt, nf))   # This is the spectrogram calculated in VocSectioningAmp
    
    # Create BioSound Object and store some values
    myBioSound = BioSound(soundWave = soundIn, fs = samprate, emitter=birdNameCutsStrs[inds], calltype = vocTypeCutsStrs[inds])
    myBioSound.spectro = soundSpect    # Log spectrogram
    myBioSound.to = to         # Time scale for spectrogram
    myBioSound.fo = fo         # Frequency scale for spectrogram
    myBioSound.rms = rms       # The rms
       
    # Calculate amplitude enveloppe
    myBioSound.ampenv()
       
    # Calculate the power spectrum
    myBioSound.spectrum(f_high=10000)
    
    # Calculate fundamental and related values
    myBioSound.fundest()
      
    if plotme:  
        myBioSound.plot()  
        myBioSound.play()
        waitstr = raw_input("Press Enter to continue...")
        plt.close('all')

    # Save the results
    fname = '%s_%s_%d.h5' % (myBioSound.emitter, myBioSound.type, inds)
    myBioSound.saveh5(fname)

