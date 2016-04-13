import pandas
import numpy as np
import matplotlib.pyplot as plt
from lasp.discriminate import discriminatePlot


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
ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses = discriminatePlot(X.transpose(), y, cVal, titleStr='Call Type Sf_F')
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
ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses = discriminatePlot(X.transpose()[noSoInd], y[noSoInd], cVal[noSoInd], titleStr='Call Type NoSo Sf_F')
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

ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses = discriminatePlot(X.transpose(), y, cVal, titleStr='Call Type Sf')
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

ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses = discriminatePlot(X.transpose()[noSoInd], y[noSoInd], cVal[noSoInd], titleStr='Call Type NoSo Sf')
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

ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses = discriminatePlot(X.transpose(), y, cVal, titleStr='Call Type F')
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

ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses = discriminatePlot(X.transpose()[noSoInd], y[noSoInd], cVal[noSoInd], titleStr='Call Type NoSo F')
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
ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses = discriminatePlot(X[adultInd], y[adultInd], cValBirdAll[adultInd], titleStr='Caller Sf_F')
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
ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses = discriminatePlot(X[adultInd], y[adultInd], cValBirdAll[adultInd], titleStr='Caller Sf')
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

ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses = discriminatePlot(X[adultInd], y[adultInd], cValBirdAll[adultInd], titleStr='Caller F')
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
        lda, ldaSE, qda, qdaSE, rf, rfSE, nC = discriminatePlot(X[goodInd], y[goodInd], cValBirdAll[goodInd], titleStr='Voice for %s' % ctype)
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