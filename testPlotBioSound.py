from lasp.sound import BioSound 
import os

# Read the matlab file that has all the cut sounds
os.chdir('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/h5files')

myBioSound = BioSound()
myBioSound.readh5('YelOra2575_Tu_7945.h5')
myBioSound.plot()
myBioSound.play()