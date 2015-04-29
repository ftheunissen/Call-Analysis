from lasp.sound import BioSound 
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.ensemble import RandomForestClassifier as RF

def ldaPlot(X, y, cVal, titleStr=''):
    
    # Initialize
    ldaMod = LDA(n_components = 2) 
    qdaMod = QDA()
    rfMod = RF()
    
    # Fit
    Xr = ldaMod.fit_transform(X, y)
    qdaMod.fit(X, y)
    rfMod.fit(X,y)
    
    # Predict      
    yPredLDA = ldaMod.predict(X) 
    pCLDA = float(sum(y==yPredLDA))/len(y)
    yPredQDA = qdaMod.predict(X) 
    pCQDA = float(sum(y==yPredQDA))/len(y)
    yPredRF = rfMod.predict(X) 
    pCRF = float(sum(y==yPredRF))/len(y)
    
    # Plot       
    plt.figure()
    plt.scatter(Xr[:,0], Xr[:,1], c=cVal, s=40)
    plt.title('%s: pC %.2f %% %.2f %% %.2f %%' % (titleStr, (pCLDA*100.0), (pCQDA*100.0), (pCRF*100.0)))
    plt.xlabel('DFA 1')
    plt.ylabel('DFA 2')  
    plt.show()
    return

# Read the matlab file that has all the cut sounds
os.chdir('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/h5files')

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
vocSelTableGrouped = vocSelTable.groupby(['Bird','calltype'])
vocSelTableGroupedAgg = vocSelTableGrouped.aggregate('mean').reset_index()


# Plot some nice graphs
callColor = {'Be': (0/255.0, 230/255.0, 255/255.0), 'LT': (0/255.0, 95/255.0, 255/255.0), 'Tu': (255/255.0, 200/255.0, 65/255.0), 'Th': (255/255.0, 150/255.0, 40/255.0), 
             'Di': (255/255.0, 105/255.0, 15/255.0), 'Ag': (255/255.0, 0/255.0, 0/255.0), 'Wh': (255/255.0, 180/255.0, 255/255.0), 'Ne': (255/255.0, 100/255.0, 255/255.0),
             'Te': (140/255.0, 100/255.0, 185/255.0), 'DC': (100/255.0, 50/255.0, 200/255.0), 'So': (255/255.0, 255/255.0, 255/255.0)}

cVal = []
for cType in vocSelTableGroupedAgg['calltype']:
    cVal.append(callColor[cType])

# Display formants
plt.figure()
ax = plt.subplot(131)
plt.scatter(vocSelTableGroupedAgg['F1'], vocSelTableGroupedAgg['F2'], c=cVal, s=40)
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('F1 (Hz)')
plt.ylabel('F2 (Hz)')
plt.xlim((1000, 6000))
plt.ylim((1000, 6000))
ax = plt.subplot(132)
plt.scatter(vocSelTableGroupedAgg['F1'], vocSelTableGroupedAgg['F3'], c=cVal, s=40)
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('F1 (Hz)')
plt.ylabel('F3 (Hz)')
plt.xlim((1000, 6000))
plt.ylim((1000, 6000))
ax = plt.subplot(133)
plt.scatter(vocSelTableGroupedAgg['F2'], vocSelTableGroupedAgg['F3'], c=cVal, s=40)
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('F2 (Hz)')
plt.ylabel('F3 (Hz)')
plt.xlim((1000, 6000))
plt.ylim((1000, 6000))
plt.show()

# Display Saliency vs Formants
plt.figure()
plt.subplot(131)
plt.scatter(vocSelTableGroupedAgg['sal'], vocSelTableGroupedAgg['F1'], c=cVal, s=40)
plt.xlabel('sal')
plt.ylabel('F1 (Hz)')
plt.subplot(132)
plt.scatter(vocSelTableGroupedAgg['sal'], vocSelTableGroupedAgg['F2'], c=cVal, s=40)
plt.xlabel('sal')
plt.ylabel('F2 (Hz)')
plt.subplot(133)
plt.scatter(vocSelTableGroupedAgg['sal'], vocSelTableGroupedAgg['F3'], c=cVal, s=40)
plt.xlabel('sal')
plt.ylabel('F3 (Hz)')
plt.show()


plt.figure()
plt.subplot(131)
plt.scatter(vocSelTableGroupedAgg['F1'], vocSelTableGroupedAgg['F2']-(1.0*vocSelTableGroupedAgg['F1']), c=cVal, s=40)
plt.xlabel('F1 (Hz)')
plt.ylabel('F2 - F1 (Hz)')
plt.subplot(132)
plt.scatter(vocSelTableGroupedAgg['F1'], vocSelTableGroupedAgg['F3']-(2.0*vocSelTableGroupedAgg['F1']), c=cVal, s=40)
plt.xlabel('F1 (Hz)')
plt.ylabel('F3 - 2*F1 (Hz)')
plt.subplot(133)
plt.scatter(vocSelTableGroupedAgg['F2'], vocSelTableGroupedAgg['F3']-(1.0*vocSelTableGroupedAgg['F2']), c=cVal, s=40)
plt.xlabel('F2 (Hz)')
plt.ylabel('F3 - F2 (Hz)')
plt.show()

plt.figure()
plt.subplot(131)
plt.scatter(vocSelTableGroupedAgg['q1'], vocSelTableGroupedAgg['q2'], c=cVal, s=40)
plt.xlabel('Q1 (Hz)')
plt.ylabel('Q2 (Hz)')
plt.subplot(132)
plt.scatter(vocSelTableGroupedAgg['q1'], vocSelTableGroupedAgg['q3'], c=cVal, s=40)
plt.xlabel('Q1 (Hz)')
plt.ylabel('Q3 (Hz)')
plt.subplot(133)
plt.scatter(vocSelTableGroupedAgg['q2'], vocSelTableGroupedAgg['q3'], c=cVal, s=40)
plt.xlabel('Q2 (Hz)')
plt.ylabel('Q3 (Hz)')
plt.show()

# Perform some classifications - with and without song
X = np.array([vocSelTableGroupedAgg['F1'], vocSelTableGroupedAgg['F2'], vocSelTableGroupedAgg['F3'], vocSelTableGroupedAgg['sal']])
y = np.array(vocSelTableGroupedAgg['calltype'])
noSoInd = (y != 'So')
cVal = np.asarray(cVal)
ldaPlot(X.transpose(), y, cVal, titleStr='F1-3')
ldaPlot(X.transpose()[noSoInd], y[noSoInd], cVal[noSoInd], titleStr='No So F1-3')

X = np.array([np.log(vocSelTableGroupedAgg['F1']), np.log(vocSelTableGroupedAgg['F2']), np.log(vocSelTableGroupedAgg['F3']), vocSelTableGroupedAgg['sal']])
ldaPlot(X.transpose(), y, cVal, titleStr='Log F1-3')
ldaPlot(X.transpose()[noSoInd], y[noSoInd], cVal[noSoInd], titleStr='No So Log F1-3')
        
X = np.array([np.log(vocSelTableGroupedAgg['q1']), np.log(vocSelTableGroupedAgg['q2']), np.log(vocSelTableGroupedAgg['q3']), vocSelTableGroupedAgg['sal']])
ldaPlot(X.transpose(), y, cVal, titleStr='Log Q1-3')
ldaPlot(X.transpose()[noSoInd], y[noSoInd], cVal[noSoInd], titleStr='No So Log Q1-3')