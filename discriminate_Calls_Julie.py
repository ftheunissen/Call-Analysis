import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import cross_validation
from scipy import stats

def ldaPlot(X, y, cVal, titleStr=''):
    # Frederic's Robust Wrapper for discriminant analysis function.  Performs lda, qda and RF and returns cross-validated
    # performance, stderr and base line.
    
    # Global Parameters
    CVFOLDS = 10
    MINCOUNT = 10
    MINCOUNTTRAINING = 5 
    
    # Initialize Variables and clean up data
    classes, classesCount = np.unique(y, return_counts = True)  # Classes to be discriminated should be same as ldaMod.classes_
    goodIndClasses = np.array([n >= MINCOUNT for n in classesCount])
    goodInd = np.array([b in classes[goodIndClasses] for b in y])
    yGood = y[goodInd]
    XGood = X[goodInd]
    cValGood = cVal[goodInd]


    classes, classesCount = np.unique(yGood, return_counts = True) 
    nClasses = classes.size         # Number of classes or groups  

    # Do we have enough data?  
    if (nClasses < 2):
        print 'Error in ldaPLot: Insufficient classes with minimun data (%d) for discrimination analysis' % (MINCOUNT)
        return -1, -1, -1, -1 , -1, -1, -1
    cvFolds = min(min(classesCount), CVFOLDS)
    if (cvFolds < CVFOLDS):
        print 'Warning in ldaPlot: Cross-validation performed with %d folds (instead of %d)' % (cvFolds, CVFOLDS)
   
    # Data size and color values   
    nD = XGood.shape[1]                 # Dimensionality of X
    cClasses = []   # Color code for each class
    for cl in classes:
        icl = (yGood == cl).nonzero()[0][0]
        cClasses.append(np.append(cValGood[icl],1.0))
    cClasses = np.asarray(cClasses)
    myPrior = np.ones(nClasses)*(1.0/nClasses)  

      
    # Initialise Classifiers  
    ldaMod = LDA(n_components = min(nD,nClasses-1), priors = myPrior ) 
    qdaMod = QDA(priors = myPrior)
    rfMod = RF()   # by default assumes equal weights

    
    # Use lda to do a dimensionality reduction in discriminant function space
    Xr = ldaMod.fit_transform(XGood, yGood) 
    
    # Check labels
    for a, b in zip(classes, ldaMod.classes_):
        if a != b:
            print 'Error in ldaPlot: labels do not match'

  
    # Print the coefficients of first 3 DFA 
    print 'LDA Weights:'
    print 'DFA1:', ldaMod.coef_[0,:]
    print 'DFA2:', ldaMod.coef_[1,:] 
    print 'DFA3:', ldaMod.coef_[2,:] 
        
    # Perform CVFOLDS fold cross-validation to get performance of classifiers.
    # Here using X or Xr should give very similar results
    ldaScores = np.zeros(cvFolds)
    qdaScores = np.zeros(cvFolds)
    rfScores = np.zeros(cvFolds)
    skf = cross_validation.StratifiedKFold(yGood, cvFolds)
    iskf = 0
    
    for train, test in skf:
        
        # Enforce the MINCOUNT in each class for Training
        trainClasses, trainCount = np.unique(yGood[train], return_counts=True)
        goodIndClasses = np.array([n >= MINCOUNTTRAINING for n in trainCount])
        goodIndTrain = np.array([b in trainClasses[goodIndClasses] for b in yGood[train]])

        # Specity the training data set, the number of groups and priors
        yTrain = yGood[train[goodIndTrain]]
        XrTrain = Xr[train[goodIndTrain]]
        trainClasses, trainCount = np.unique(yTrain, return_counts=True) 
        ntrainClasses = trainClasses.size
        
        # Skip this cross-validation fold because of insufficient data
        if ntrainClasses < 2:
            continue
        goodInd = np.array([b in trainClasses for b in yGood[test]])    
        if (goodInd.size == 0):
            continue
           
        # Fit the data
        trainPriors = np.ones(ntrainClasses)*(1.0/ntrainClasses)
        ldaMod.priors = trainPriors
        qdaMod.priors = trainPriors
        ldaMod.fit(XrTrain, yTrain)
        qdaMod.fit(XrTrain, yTrain)        
        rfMod.fit(XrTrain, yTrain)
        

        ldaScores[iskf] = ldaMod.score(Xr[test[goodInd]], yGood[test[goodInd]])
        qdaScores[iskf] = qdaMod.score(Xr[test[goodInd]], yGood[test[goodInd]])
        rfScores[iskf] = rfMod.score(Xr[test[goodInd]], yGood[test[goodInd]])
        iskf += 1
     
    if (iskf !=  cvFolds):
        cvFolds = iskf
        ldaScores.reshape(cvFolds)
        qdaScores.reshape(cvFolds)
        rfScores.reshape(cvFolds)
                 
    ldaMod.priors = myPrior
    qdaMod.priors = myPrior
                
    # Make a mesh for plotting
    x1, x2 = np.meshgrid(np.arange(-6.0, 6.0, 0.1), np.arange(-6.0, 6.0, 0.1))
    xm1 = np.reshape(x1, -1)
    xm2 = np.reshape(x2, -1)
    nxm = np.size(xm1)
    Xm = np.zeros((nxm, 2))
    Xm[:,0] = xm1
    Xm[:,1] = xm2
    XmcLDA = np.zeros((nxm, 4))  # RGBA values for color for LDA
    XmcQDA = np.zeros((nxm, 4))  # RGBA values for color for QDA
    XmcRF = np.zeros((nxm, 4))  # RGBA values for color for RF

    
    # Predict values on mesh for plotting based on the first two DFs  
    ldaMod.fit(Xr[:, 0:2], yGood)    
    qdaMod.fit(Xr[:, 0:2], yGood)
    rfMod.fit(Xr[:, 0:2], yGood) 
    yPredLDA = ldaMod.predict_proba(Xm) 
    yPredQDA = qdaMod.predict_proba(Xm) 
    yPredRF = rfMod.predict_proba(Xm)
    
    # Transform the predictions in color codes
    maxLDA = yPredLDA.max()
    for ix in range(nxm) :
        cWeight = yPredLDA[ix,:]                               # Prob for all classes
        cWinner = ((cWeight == cWeight.max()).astype('float')) # Winner takes all 
        # XmcLDA[ix,:] = np.dot(cWeight, cClasses)/nClasses
        XmcLDA[ix,:] = np.dot(cWinner, cClasses)
        XmcLDA[ix,3] = cWeight.max()/maxLDA
    
    # Plot the surface of probability    
    plt.figure(facecolor='white', figsize=(10,3))
    plt.subplot(131)
    Zplot = XmcLDA.reshape(np.shape(x1)[0], np.shape(x1)[1],4)
    plt.imshow(Zplot, zorder=0, extent=[-6, 6, -6, 6], origin='lower', interpolation='none', aspect='auto')
    plt.scatter(Xr[:,0], Xr[:,1], c=cValGood, s=40, zorder=1)
    plt.title('%s: LDA pC %.2f %%' % (titleStr, (ldaScores.mean()*100.0)))
    plt.xlim((-6, 6))
    plt.ylim((-6, 6))
    plt.axis('square')
    plt.xlabel('DFA 1')
    plt.ylabel('DFA 2')

    
    # Transform the predictions in color codes
    maxQDA = yPredQDA.max()
    for ix in range(nxm) :
        cWeight = yPredQDA[ix,:]                               # Prob for all classes
        cWinner = ((cWeight == cWeight.max()).astype('float')) # Winner takes all 
        # XmcLDA[ix,:] = np.dot(cWeight, cClasses)/nClasses
        XmcQDA[ix,:] = np.dot(cWinner, cClasses)
        XmcQDA[ix,3] = cWeight.max()/maxQDA
    
    # Plot the surface of probability    
    plt.subplot(132)
    Zplot = XmcQDA.reshape(np.shape(x1)[0], np.shape(x1)[1],4)
    plt.imshow(Zplot, zorder=0, extent=[-6, 6, -6, 6], origin='lower', interpolation='none', aspect='auto')
    plt.scatter(Xr[:,0], Xr[:,1], c=cValGood, s=40, zorder=1)
    plt.title('%s: QDA pC %.2f %%' % (titleStr, (qdaScores.mean()*100.0)))
    plt.xlabel('DFA 1')
    plt.ylabel('DFA 2')
    plt.xlim((-6, 6))
    plt.ylim((-6, 6))
    plt.axis('square')
    
    # Transform the predictions in color codes
    maxRF = yPredRF.max()
    for ix in range(nxm) :
        cWeight = yPredRF[ix,:]           # Prob for all classes
        cWinner = ((cWeight == cWeight.max()).astype('float')) # Winner takes all 
        # XmcLDA[ix,:] = np.dot(cWeight, cClasses)/nClasses  # Weighted colors does not work
        XmcRF[ix,:] = np.dot(cWinner, cClasses)
        XmcRF[ix,3] = cWeight.max()/maxRF
    
    # Plot the surface of probability    
    plt.subplot(133)
    Zplot = XmcRF.reshape(np.shape(x1)[0], np.shape(x1)[1],4)
    plt.imshow(Zplot, zorder=0, extent=[-6, 6, -6, 6], origin='lower', interpolation='none', aspect='auto')
    plt.scatter(Xr[:,0], Xr[:,1], c=cValGood, s=40, zorder=1)
    plt.title('%s: RF pC %.2f %%' % (titleStr, (rfScores.mean()*100.0)))
    plt.xlabel('DFA 1')
    plt.ylabel('DFA 2')
    plt.xlim((-6, 6))
    plt.ylim((-6, 6))
    plt.axis('square')
    plt.show()


    # Results
    ldaScore = ldaScores.mean()*100.0
    qdaScore = qdaScores.mean()*100.0
    rfScore = rfScores.mean()*100.0
    ldaScoreSE = ldaScores.std() * 100.0
    qdaScoreSE = qdaScores.std() * 100.0 
    rfScoreSE = rfScores.std() * 100.0 
    
    print ("Number of classes %d. Chance level %.2f %%") % (nClasses, 100.0/nClasses)
    print ("%s LDA: %.2f (+/- %0.2f) %%") % (titleStr, ldaScore, ldaScoreSE)
    print ("%s QDA: %.2f (+/- %0.2f) %%") % (titleStr, qdaScore, qdaScoreSE)
    print ("%s RF: %.2f (+/- %0.2f) %%") % (titleStr, rfScore, rfScoreSE)
    return ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses

# Read the hdf file that has all the acoustical parameters of the entire call database
# These parameters are extracted from makeDataFrame_Calls_Julie

store = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/vocSelTable.h5'
figdir = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/Figures Voice'
vocSelTable = pandas.read_hdf(store)
vocSelTableGrouped = vocSelTable.groupby(['Bird','calltype'])
vocSelTableGroupedAgg = vocSelTableGrouped.aggregate('mean').reset_index()


# Color code for call type
callColor = {'Be': (0/255.0, 230/255.0, 255/255.0), 'LT': (0/255.0, 95/255.0, 255/255.0), 'Tu': (255/255.0, 200/255.0, 65/255.0), 'Th': (255/255.0, 150/255.0, 40/255.0), 
             'Di': (255/255.0, 105/255.0, 15/255.0), 'Ag': (255/255.0, 0/255.0, 0/255.0), 'Wh': (255/255.0, 180/255.0, 255/255.0), 'Ne': (255/255.0, 100/255.0, 255/255.0),
             'Te': (140/255.0, 100/255.0, 185/255.0), 'DC': (100/255.0, 50/255.0, 200/255.0), 'So': (255/255.0, 255/255.0, 255/255.0)}

callTypes = np.unique(vocSelTableGroupedAgg['calltype'])

cVal = []
for cType in vocSelTableGroupedAgg['calltype']:
    cVal.append(callColor[cType])
    
# Color code for bird ID
birdNames = np.unique(vocSelTableGroupedAgg['Bird'])

birdColor = {}
for birdId in birdNames:
    birdColor[birdId] = np.random.rand(3)
    
cValBirdAll = []
for birdId in vocSelTable['Bird']:
    cValBirdAll.append(birdColor[birdId])
    
cValBird = []
for birdId in vocSelTableGroupedAgg['Bird']:
    cValBird.append(birdColor[birdId])

# Perform some classifications - with and without song - these are averaged per bird.
X = np.array([vocSelTableGroupedAgg['F1'], vocSelTableGroupedAgg['F2'], vocSelTableGroupedAgg['F3'], vocSelTableGroupedAgg['sal'], vocSelTableGroupedAgg['fund'], vocSelTableGroupedAgg['cvfund']])
y = np.array(vocSelTableGroupedAgg['calltype'])
noSoInd = (y != 'So')
cVal = np.asarray(cVal)

# Spectral + Fund Model
ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses = ldaPlot(X.transpose(), y, cVal, titleStr='Call Type Sf_F')
d = {'Type': np.array(['Rep']),
     'Features': np.array(['Sf_F']), 
     'LDA' : np.array([ldaScore]), 
     'QDA': np.array([qdaScore]),
     'RF': np.array([rfScore]), 
     'nClasses' : np.array([nClasses]),
     'LDA_SE' : np.array([ldaScoreSE]),
     'QDA_SE': np.array([qdaScoreSE]),
     'RF_SE': np.array([rfScoreSE])}
     
resultsDataFrame= pandas.DataFrame(data = d)
plt.savefig('%s/Call Type Sf_F.eps' % figdir)
 
# Repeat without song 
ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses = ldaPlot(X.transpose()[noSoInd], y[noSoInd], cVal[noSoInd], titleStr='Call Type NoSo Sf_F')
d = {'Type': np.array(['RepnoSo']),
     'Features': np.array(['Sf_F']), 
     'LDA' : np.array([ldaScore]), 
     'QDA': np.array([qdaScore]),
     'RF': np.array([rfScore]), 
     'nClasses' : np.array([nClasses]),
     'LDA_SE' : np.array([ldaScoreSE]),
     'QDA_SE': np.array([qdaScoreSE]),
     'RF_SE': np.array([rfScoreSE])}
     
tempdf = pandas.DataFrame(data=d)
resultsDataFrame = pandas.concat([resultsDataFrame, tempdf], ignore_index = True )
plt.savefig('%s/Call Type NoSo Sf_F.eps' % figdir)

X = np.array([vocSelTableGroupedAgg['F1'], vocSelTableGroupedAgg['F2'], vocSelTableGroupedAgg['F3'], vocSelTableGroupedAgg['sal']])

ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses = ldaPlot(X.transpose(), y, cVal, titleStr='Call Type Sf')
d = {'Type': np.array(['Rep']),
     'Features': np.array(['Sf']), 
     'LDA' : np.array([ldaScore]), 
     'QDA': np.array([qdaScore]),
     'RF': np.array([rfScore]), 
     'nClasses' : np.array([nClasses]),
     'LDA_SE' : np.array([ldaScoreSE]),
     'QDA_SE': np.array([qdaScoreSE]),
     'RF_SE': np.array([rfScoreSE])}
tempdf = pandas.DataFrame(data=d)
resultsDataFrame = pandas.concat([resultsDataFrame, tempdf], ignore_index = True )
plt.savefig('%s/Call Type Sf.eps' % figdir)

ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses = ldaPlot(X.transpose()[noSoInd], y[noSoInd], cVal[noSoInd], titleStr='Call Type NoSo Sf')
d = {'Type': np.array(['RepnoSo']),
     'Features': np.array(['Sf']), 
     'LDA' : np.array([ldaScore]), 
     'QDA': np.array([qdaScore]),
     'RF': np.array([rfScore]), 
     'nClasses' : np.array([nClasses]),
     'LDA_SE' : np.array([ldaScoreSE]),
     'QDA_SE': np.array([qdaScoreSE]),
     'RF_SE': np.array([rfScoreSE])}
     
tempdf = pandas.DataFrame(data=d)
resultsDataFrame = pandas.concat([resultsDataFrame, tempdf], ignore_index = True )
plt.savefig('%s/Call Type NoSo Sf.eps' % figdir)


X = np.array([np.log(vocSelTableGroupedAgg['fund']), np.log(vocSelTableGroupedAgg['cvfund']), vocSelTableGroupedAgg['sal']])

ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses = ldaPlot(X.transpose(), y, cVal, titleStr='Call Type F')
d = {'Type': np.array(['Rep']),
     'Features': np.array(['F']), 
     'LDA' : np.array([ldaScore]), 
     'QDA': np.array([qdaScore]),
     'RF': np.array([rfScore]), 
     'nClasses' : np.array([nClasses]),
     'LDA_SE' : np.array([ldaScoreSE]),
     'QDA_SE': np.array([qdaScoreSE]),
     'RF_SE': np.array([rfScoreSE])}
tempdf = pandas.DataFrame(data=d)
resultsDataFrame = pandas.concat([resultsDataFrame, tempdf], ignore_index = True )
plt.savefig('%s/Call Type F.eps' % figdir)

ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses = ldaPlot(X.transpose()[noSoInd], y[noSoInd], cVal[noSoInd], titleStr='Call Type NoSo F')
d = {'Type': np.array(['RepnoSo']),
     'Features': np.array(['F']), 
     'LDA' : np.array([ldaScore]), 
     'QDA': np.array([qdaScore]),
     'RF': np.array([rfScore]), 
     'nClasses' : np.array([nClasses]),
     'LDA_SE' : np.array([ldaScoreSE]),
     'QDA_SE': np.array([qdaScoreSE]),
     'RF_SE': np.array([rfScoreSE])}
     
tempdf = pandas.DataFrame(data=d)
resultsDataFrame = pandas.concat([resultsDataFrame, tempdf], ignore_index = True )
plt.savefig('%s/Call Type NoSo F.eps' % figdir)


# Perform the discrimination for birds

# Perform some classifications - for adult birds - these are averaged per bird.

cValBirdAll = np.asarray(cValBirdAll)
y = np.array(vocSelTable['calltype'])

X = np.array([vocSelTable['F1'], vocSelTable['F2'], vocSelTable['F3'], vocSelTable['sal'], vocSelTable['fund'], vocSelTable['cvfund'] ])
X = X.transpose()
nonanInd = (np.sum(np.isnan(X), axis = 1) == 0)
adultInd = (y != 'Be') & (y!= 'LT') & nonanInd
y = np.array(vocSelTable['Bird'])

# Spectral + Fund Model
ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses = ldaPlot(X[adultInd], y[adultInd], cValBirdAll[adultInd], titleStr='Caller Sf_F')
d = {'Type': np.array(['CallerAdult']),
     'Features': np.array(['Sf_F']), 
     'LDA' : np.array([ldaScore]), 
     'QDA': np.array([qdaScore]),
     'RF': np.array([rfScore]), 
     'nClasses' : np.array([nClasses]),
     'LDA_SE' : np.array([ldaScoreSE]),
     'QDA_SE': np.array([qdaScoreSE]),
     'RF_SE': np.array([rfScoreSE])}
     
tempdf = pandas.DataFrame(data = d)
resultsDataFrame = pandas.concat([resultsDataFrame, tempdf], ignore_index = True )
plt.savefig('%s/Caller Sf_F.eps' % figdir)

X = np.array([vocSelTable['F1'], vocSelTable['F2'], vocSelTable['F3'], vocSelTable['sal']])
X = X.transpose()
ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses = ldaPlot(X[adultInd], y[adultInd], cValBirdAll[adultInd], titleStr='Caller Sf')
d = {'Type': np.array(['CallerAdult']),
     'Features': np.array(['Sf']), 
     'LDA' : np.array([ldaScore]), 
     'QDA': np.array([qdaScore]),
     'RF': np.array([rfScore]), 
     'nClasses' : np.array([nClasses]),
     'LDA_SE' : np.array([ldaScoreSE]),
     'QDA_SE': np.array([qdaScoreSE]),
     'RF_SE': np.array([rfScoreSE])}
tempdf = pandas.DataFrame(data=d)
resultsDataFrame = pandas.concat([resultsDataFrame, tempdf], ignore_index = True )
plt.savefig('%s/Caller Sf.eps' % figdir)


X = np.array([vocSelTable['fund'], vocSelTable['cvfund'], vocSelTable['sal']])
X = X.transpose()

ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses = ldaPlot(X[adultInd], y[adultInd], cValBirdAll[adultInd], titleStr='Caller F')
d = {'Type': np.array(['CallerAdult']),
     'Features': np.array(['F']), 
     'LDA' : np.array([ldaScore]), 
     'QDA': np.array([qdaScore]),
     'RF': np.array([rfScore]), 
     'nClasses' : np.array([nClasses]),
     'LDA_SE' : np.array([ldaScoreSE]),
     'QDA_SE': np.array([qdaScoreSE]),
     'RF_SE': np.array([rfScoreSE])}
tempdf = pandas.DataFrame(data=d)
resultsDataFrame = pandas.concat([resultsDataFrame, tempdf], ignore_index = True )
plt.savefig('%s/Caller F.eps' % figdir)


# Loop through call types:
cValBird = np.asarray(cValBird)
cValBirdAll = np.asarray(cValBirdAll)
y = np.array(vocSelTable['Bird'])
X = np.array([vocSelTable['F1'], vocSelTable['F2'], vocSelTable['F3'], vocSelTable['sal'], vocSelTable['fund'], vocSelTable['cvfund'] ])
X = X.transpose()
nonanInd = (np.sum(np.isnan(X), axis = 1) == 0)
ldaScore = []
ldaScoreSE = []
qdaScore = []
qdaScoreSE = []
rfScore = [] 
rfScoreSE = []
nClasses = []
Type = []
Features = []

for ctype in callTypes:
    goodInd = (vocSelTable['calltype'] == ctype) & nonanInd
    goodInd = goodInd.nonzero()
    nsamples = np.size(goodInd)
    nbirds = np.unique(y[goodInd]).size
    print 'Performing Voice Discrimination for %s with %d samples from %d birds' % (ctype, nsamples, nbirds)
    if (nbirds < 2) :
        print 'Warning: Insuficient number of birds to perform discrimination'
    else:
        lda, ldaSE, qda, qdaSE, rf, rfSE, nC = ldaPlot(X[goodInd], y[goodInd], cValBirdAll[goodInd], titleStr='Voice for %s' % ctype)
        if lda == -1:
            continue
        ldaScore.append(lda)
        ldaScoreSE.append(ldaSE)
        qdaScore.append(qda)
        qdaScoreSE.append(qdaSE)
        rfScore.append(rf) 
        rfScoreSE.append(rfSE)
        nClasses.append(nC)
        Features.append('SF_f')
        Type.append('Caller%s' % ctype)
        plt.savefig('%s/Caller%s.eps' % (figdir, ctype))
        
d = {'Type': np.array(Type),
     'Features': np.array(Features), 
     'LDA' : np.array(ldaScore), 
     'QDA': np.array(qdaScore),
     'RF': np.array(rfScore), 
     'nClasses' : np.array(nClasses),
     'LDA_SE' : np.array(ldaScoreSE),
     'QDA_SE': np.array(qdaScoreSE),
     'RF_SE': np.array(rfScoreSE)}
     
tempdf = pandas.DataFrame(data=d)
resultsDataFrame = pandas.concat([resultsDataFrame, tempdf], ignore_index = True )