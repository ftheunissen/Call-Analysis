from lasp.sound import BioSound, mps, plot_mps
import os
import numpy as np
import matplotlib.pyplot as plt


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
        
        wf, wt, mps_powAvg = mps(myBioSound.spectro, myBioSound.fo, myBioSound.to, 1.0)
        
        plot_mps(wf, wt, 10*np.log10(mps_powAvg))
        plt.pause(5)
        
        