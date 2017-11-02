# Calculates mixed effect glm on Voice.
library(lme4)
library(effects)
library(car)

# Currently set up for RF - do a global search to change to LDA if needed.

# Read Excel data file.
fileCSVTable = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossDiscrimResults.csv'
vocSelTable <- read.csv(fileCSVTable, header = TRUE)


# Set up the output files
fileModelCoef = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossGLMModelRF.csv'
fileModelCoefSpect = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossGLMModelSpectRF.csv'
fileModelCoefTemp = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossGLMModelTempRF.csv'
fileModelCoefFund = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossGLMModelFundRF.csv'

fileModel = as.list(c(fileModelCoef, fileModelCoefSpect, fileModelCoefTemp, fileModelCoefFund))
features = as.list(c('18 AF', 'Spect AF', 'Temp AF', 'Fund AF'))

for (ifeat in 1:length(features)) {
  ndata <- sum(vocSelTable$Features == features[[ifeat]])
  print(sprintf('N=%d for %s', ndata, features[[ifeat]]))

# First the model for the voice only: from training on other categories and fitting on one
# Note that this data only has adult calls because 
  model.Voice <- glmer( cbind(RFYes, Count-RFYes) ~ Call + (1| BirdPair) , data=vocSelTable, subset = (Features == features[[ifeat]] & TestType =='Voice'), family= binomial)
  model.Null <- glmer( cbind(RFYes, Count-RFYes) ~ 1 + (1| BirdPair) , data=vocSelTable, subset = (Features == features[[ifeat]] & TestType =='Voice') , family= binomial)
  anova(model.Null, model.Voice, test = 'Chisq')

  model.Voice.Effect <- effect("Call", model.Voice, se=TRUE)
  sum.Voice.Effect <- summary(model.Voice.Effect)
  model.Voice.Table <- cbind(Effect=sum.Voice.Effect$effect, LB=sum.Voice.Effect$lower, UB=sum.Voice.Effect$upper)

  model.CallerVoice <- glmer( cbind(RFYes, Count-RFYes) ~ Call*TestType + (1| BirdPair) , data=vocSelTable, subset = (Features == features[[ifeat]]), family= binomial)
  (summary(model.CallerVoice))

  model.CallerVoice.Effect <- effect("Call:TestType", model.CallerVoice, se=TRUE)
  sum.CallerVoice.Effect <- summary(model.CallerVoice.Effect)
  model.CallerVoice.Table <- cbind(Effect=sum.CallerVoice.Effect$effect, LB=sum.CallerVoice.Effect$lower, UB=sum.CallerVoice.Effect$upper)

  plot(model.CallerVoice)
  plot(model.CallerVoice.Effect)

  write.csv(model.CallerVoice.Table, file = fileModel[[ifeat]])

}
