# Calculates mixed effect glm on Voice.
# This is the version that performs the model for all features but does NOT (yet) do the model.
library(lme4)
library(effects)
library(car)

# Read Excel data file 
fileCSVTable = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerDiscrimResults.csv'
vocSelTable <- read.csv(fileCSVTable, header = TRUE)

# Set up the output files
fileModelCoef = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerDiscrimGLMModel.csv'
fileModelCoefSpect = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerDiscrimGLMModelSpect.csv'
fileModelCoefTemp = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerDiscrimGLMModelTemp.csv'
fileModelCoefFund = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerDiscrimGLMModelFund.csv'

fileModel = as.list(c(fileModelCoef, fileModelCoefSpect, fileModelCoefTemp, fileModelCoefFund))
features = as.list(c('18 AF', 'Spect AF', 'Temp AF', 'Fund AF'))

for (ifeat in 1:length(features)) {
  ndata <- sum(vocSelTable$Features == features[[ifeat]])
  print(sprintf('N=%d for %s', ndata, features[[ifeat]]))

# This number must correspond to the number of cross validations

  model.Voice <- glmer( cbind(LDAYes, Count-LDAYes) ~ Type + (1| BirdPair) , data=vocSelTable, subset = Features == features[[ifeat]], family= binomial)
  model.Null <- glmer( cbind(LDAYes, Count-LDAYes) ~ 1 + (1| BirdPair) , data=vocSelTable, subset = Features == features[[ifeat]], family= binomial)

  print(anova(model.Null, model.Voice, test = 'Chisq'))

  model.Effect <- effect("Type", model.Voice, se=TRUE)
  plot(model.Effect)
  sum.Effect <- summary(model.Effect)

  model.Table <- cbind(Effect=sum.Effect$effect, LB=sum.Effect$lower, UB=sum.Effect$upper)

  write.csv(model.Table, file = fileModel[[ifeat]])

}

