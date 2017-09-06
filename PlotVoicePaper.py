from soundsig.sound import BioSound 
import os
import matplotlib.pyplot as plt



# Read the matlab file that has all the cut Sounds
os.chdir('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/h5files')
figdir = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/Figures Voice'

callTypes = ['Ag', 'DC']
#birds = ['GreOra1817F', 'WhiRas44ddF']
#birdFiles = {'GreOra1817F': [['GreOra1817_Ag_2215.h5', 'GreOra1817_Ag_2239.h5', 'GreOra1817_Ag_2244.h5'],
#               ['GreOra1817_DC_2267.h5', 'GreOra1817_DC_2275.h5', 'GreOra1817_DC_2285.h5']],
#             'WhiRas44ddF':  [['WhiRas44dd_Ag_6841.h5', 'WhiRas44dd_Ag_6919.h5', 'WhiRas44dd_Ag_6948.h5'],
#               ['WhiRas44dd_DC_6875.h5', 'WhiRas44dd_DC_6882.h5', 'WhiRas44dd_DC_6895.h5'] ] }

birds = ['HpiHpi4748M', 'LblRed0613M']
birdFiles = {'HpiHpi4748M': [['HPiHPi4748_Ag_2414.h5', 'HPiHPi4748_Ag_2415.h5', 'HPiHPi4748_Ag_2416.h5'],
               ['HPiHPi4748_DC_2411.h5', 'HPiHPi4748_DC_2411.h5', 'HPiHPi4748_DC_2411.h5']],
             'LblRed0613M':  [['LblRed0613_Ag_5093.h5', 'LblRed0613_Ag_5094.h5', 'LblRed0613_Ag_5095.h5'],
               ['LblRed0613_DC_5763.h5', 'LblRed0613_DC_5764.h5', 'LblRed0613_DC_5765.h5']] }


# Make one plot per bird

ibird = 1
for bird in birds:
    plt.figure(ibird)  
    icall = 0
    for calls in callTypes:
        for isample in range(3):             
            myBioSound = BioSound()
            myBioSound.readh5(birdFiles[bird][icall][isample])
            myBioSound.plot()
            plt.figure(1)
            plt.savefig('%s/Spectro%s_%s_%d.eps' % (figdir, bird, calls, isample))
            
        icall +=1
    raw_input('Press Enter for next Bird--> ')
    ibird +=1
            

