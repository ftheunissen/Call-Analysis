from lasp.sound import BioSound 
from lasp.sound import WavFile
import os
import matplotlib.pyplot as plt
import numpy as np


# Read the matlab file that has all the cut Sounds
os.chdir('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/h5files')
figdir = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/Figures Voice'

#  Bird 1  DC 
silence = np.zeros(10000)
totDCSound = np.zeros(5000)
totAgSound = np.zeros(5000)
myBioSound = BioSound()
myBioSound.readh5('LblBla4419_DC_2962.h5')
myBioSound.plot()
plt.figure(1)
plt.savefig('%s/DC1_1.eps' % figdir)
totDCSound = np.append(totDCSound, myBioSound.sound)
totDCSound = np.append(totDCSound, silence)

myBioSound.readh5('LblBla4419_DC_2963.h5')
myBioSound.plot()
plt.figure(1)
plt.savefig('%s/DC1_2.eps' % figdir)
totDCSound = np.append(totDCSound, myBioSound.sound)
totDCSound = np.append(totDCSound, silence)

myBioSound.readh5('LblBla4419_DC_2964.h5')
myBioSound.plot()
plt.figure(1)
plt.savefig('%s/DC1_3.eps' % figdir)
totDCSound = np.append(totDCSound, myBioSound.sound)
totDCSound = np.append(totDCSound, silence)
totDCSound = np.append(totDCSound, silence)

# Bird 1 AG

myBioSound.readh5('LblBla4419_Ag_2927.h5')
myBioSound.plot()
plt.figure(1)
plt.savefig('%s/Ag1_1.eps' % figdir)
totAgSound = np.append(totAgSound, myBioSound.sound)
totAgSound = np.append(totAgSound, silence)

myBioSound.readh5('LblBla4419_Ag_2928.h5')
myBioSound.plot()
plt.figure(1)
plt.savefig('%s/Ag1_2.eps' % figdir)
totAgSound = np.append(totAgSound, myBioSound.sound)
totAgSound = np.append(totAgSound, silence)

myBioSound.readh5('LblBla4419_Ag_3167.h5')
myBioSound.plot()
plt.figure(1)
plt.savefig('%s/Ag1_3.eps' % figdir)
totAgSound = np.append(totAgSound, myBioSound.sound)
totAgSound = np.append(totAgSound, silence)
totAgSound = np.append(totAgSound, silence)

# Bird 2
myBioSound.readh5('YelOra2575_DC_7668.h5')
myBioSound.plot()
plt.figure(1)
plt.savefig('%s/DC2_1.eps' % figdir)
totDCSound = np.append(totDCSound, myBioSound.sound)
totDCSound = np.append(totDCSound, silence)


myBioSound.readh5('YelOra2575_DC_7669.h5')
myBioSound.plot()
plt.figure(1)
plt.savefig('%s/DC2_2.eps' % figdir)
totDCSound = np.append(totDCSound, myBioSound.sound)
totDCSound = np.append(totDCSound, silence)


myBioSound.readh5('YelOra2575_DC_7670.h5')
myBioSound.plot()
plt.figure(1)
plt.savefig('%s/DC2_3.eps' % figdir)
totDCSound = np.append(totDCSound, myBioSound.sound)
totDCSound = np.append(totDCSound, silence)

myBioSound.readh5('YelOra2575_Ag_7579.h5')
myBioSound.plot()
plt.figure(1)
plt.savefig('%s/AG2_1.eps' % figdir)
totAgSound = np.append(totAgSound, myBioSound.sound)
totAgSound = np.append(totAgSound, silence)


myBioSound.readh5('YelOra2575_Ag_7923.h5')
myBioSound.plot()
plt.figure(1)
plt.savefig('%s/AG2_2.eps' % figdir)
totAgSound = np.append(totAgSound, myBioSound.sound)
totAgSound = np.append(totAgSound, silence)


myBioSound.readh5('YelOra2575_Ag_7937.h5')
myBioSound.plot()
plt.figure(1)
plt.savefig('%s/AG2_3.eps' % figdir)
totAgSound = np.append(totAgSound, myBioSound.sound)
totAgSound = np.append(totAgSound, silence)

# myBioSound.play()

# Write the two wav files
soundWavOut = WavFile()
soundWavOut.sample_depth = 2  # in bytes
soundWavOut.sample_rate = myBioSound.samprate  # in Hz
soundWavOut.num_channels = 1

soundWavOut.data = totDCSound
soundWavOut.to_wav('%s/DCSounds.wav' % figdir, normalize=True)

soundWavOut.data = totAgSound
soundWavOut.to_wav('%s/AgSounds.wav' % figdir, normalize=True)