from soundsig.sound import BioSound, WavFile, play_wavfile 
import os
import numpy as np
import fnmatch
import matplotlib.pyplot as plt
from scipy.signal import resample

# Specify the directory that has the h5 files for all cut vocalizations
os.chdir('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/h5files')

wavdir = '/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank'

# nfiles = len([fname for fname in os.listdir('.') if fname.endswith('h5')])

# Read the bird information file
# birdInfo = np.genfromtxt('/Users/ellenzippi/Documents/TheunissenLab/data/FullVocalizationBank/Birds_List_Acoustic.txt', dtype=None);
# birdInfoNames = [ rowdata[0] for rowdata in birdInfo]

# Allocate space for fundamental, cvfund, formants, saliency, rms, spectral means, std, q1, q2, q3, time std
# Trying a list of dictionaries

# Sampling rate of wave files - this will be checked.
sampWav = 44100.0

for fname in os.listdir('.'):
    if fname.endswith('.h5'):
        myBioSound = BioSound()
        myBioSound.readh5(fname)
        Bird = np.array2string(myBioSound.emitter).translate(None, "'")
        # Clean up the data
        if Bird is 'HpiHpi4748':
            Bird = 'HPiHPi4748'
        
        callType = np.array2string(myBioSound.type).translate(None, "'")
        callTypeHD = np.array(len(callType))
        if callType =='So': 
            
            # Play and display
            myBioSound.play()
            myBioSound.plot()
            
            # Do some preprocessing for analysis           
            # First cut sound to areas of non zero
            indZeros = np.where(myBioSound.sound == 0)[0]
            
            if (indZeros.size == 0) or (indZeros[0] != 0) :
                indStart = 0
            else:
                for i in range(indZeros.size):
                    if i != indZeros[i]:
                        break;
                    indStart = i
                    
            if (indZeros.size == 0) or (indZeros[-1:] != (myBioSound.sound.size - 1)) :
                indEnd = myBioSound.sound.size-1
            else:
                k = myBioSound.sound.size - 1
                indEnd = k
                for i in range(indZeros.size-1, 0, -1):
                    if k != indZeros[i]:
                        break;
                    indEnd = k
                    k -= 1
                    
            mySound = myBioSound.sound[indStart:indEnd]
            
            # Resample if needed
            if myBioSound.samprate != sampWav:
                lensound = mySound.size
                t=(np.array(range(lensound),dtype=float))/myBioSound.samprate
                lenresampled = int(round(float(lensound)*sampWav/myBioSound.samprate))
                (mySoundWav, tresampled) = resample(mySound, lenresampled, t=t, axis=0, window=None)
            else:
                mySoundWav = mySound
                lenresampled = mySound.size
                
            # Sound power to be used for correlation coeff
            mySoundPow = np.dot(mySoundWav, mySoundWav)
                  
            # Loop through wavefiles       
            wavfiles = [fname for fname in os.listdir(wavdir) if fname.startswith(str(Bird)) if fname.startswith('Song', 18)]
            
            corrMax = 0.0
            bestFile = 0
            bestStartInd = 0
            
            for i, wavfile in enumerate(wavfiles): 
                # print(wavfile)
                # play_wavfile(os.path.join(wavdir, wavfile))
                myWavFile = WavFile(file_name=os.path.join(wavdir, wavfile))
                
                # Check expected sampling rate.
                if myWavFile.sample_rate != sampWav:
                    print('Error: Discrepancy between sampling rates')
                
                # Loop to find highest correlation
                for j in range(myWavFile.data.size-lenresampled):
                    soundTest = myWavFile.data[j:j+lenresampled]
                    soundTestPow = np.dot(soundTest, soundTest)
                    corrTest = np.dot(soundTest,mySoundWav)/np.sqrt(soundTestPow*mySoundPow)
                    if corrTest > corrMax:
                        corrMax = corrTest
                        bestFile = i
                        bestStartInd = j
                        print('Corr %.2f File %d Start %d'%(corrMax, bestFile, bestStartInd))
                
                
            # Reread best wave file for plotting
            myWavFile = WavFile(file_name=os.path.join(wavdir, wavfiles[bestFile]))
            myWavFile.plot()
            fig = plt.gcf()
            ax_list = fig.axes
            for ax in ax_list:
                ymin = ax.axis()[2]
                ymax = ax.axis()[3]
                ax.plot([myWavFile.data_t[bestStartInd],myWavFile.data_t[bestStartInd]], [ymin, ymax], color='r'  )
                ax.plot([myWavFile.data_t[bestStartInd+lenresampled],myWavFile.data_t[bestStartInd+lenresampled]], [ymin, ymax], color='r'  )
                if ax is ax_list[0]:
                    plt.title('Corr %.2f'% corrMax)

            callTypeHD = raw_input("Introductory or motif? ")
        else: 
            callTypeHD = callType

        
    