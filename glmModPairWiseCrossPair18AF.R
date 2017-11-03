# Calculates mixed effect glm on Voice.
library(lme4)
library(effects)
library(car)

# Run on this model
featureAF = '18 AF'

# Read Excel data file.
fileCSVTable = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossPairDiscrimResults.csv'
vocSelTable <- read.csv(fileCSVTable, header = TRUE)
ndata <- sum(vocSelTable$Features == featureAF)

# Output files
fileModelCoefMeanLDA = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossPairGLMModelMean18AFLDA.csv'
fileModelCoefLBLDA = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossPairGLMModelLB18AFLDA.csv'
fileModelCoefUBLDA = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossPairGLMModelUB18AFLDA.csv'

fileModelCoefMeanQDA = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossPairGLMModelMean18AFQDA.csv'
fileModelCoefLBQDA = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossPairGLMModelLB18AFQDA.csv'
fileModelCoefUBQDA = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossPairGLMModelUB18AFQDA.csv'

fileModelCoefMeanRF = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossPairGLMModelMean18AFRF.csv'
fileModelCoefLBRF = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossPairGLMModelLB18AFRF.csv'
fileModelCoefUBRF = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossPairGLMModelUB18AFRF.csv'

fileModelCoefMean = as.list(c(fileModelCoefMeanLDA, fileModelCoefMeanQDA, fileModelCoefMeanRF))
fileModelCoefLB = as.list(c(fileModelCoefLBLDA, fileModelCoefLBQDA, fileModelCoefLBRF))
fileModelCoefUB = as.list(c(fileModelCoefUBLDA, fileModelCoefUBQDA, fileModelCoefUBRF))

classifier = as.list(c('LDA', 'QDA', 'RF'))

for (ic in 1:length(classifier)) {
# Separate between Adults and Juveniles
  formula.call <- sprintf('cbind(%sYes, Count-%sYes) ~ CallTest*CallTrain + (1| BirdPair)', classifier[[ic]], classifier[[ic]])
  formula.1 <- sprintf('cbind(%sYes, Count-%sYes) ~  1+ (1| BirdPair)', classifier[[ic]], classifier[[ic]])
  
  model.CallerVoice <- glmer( as.formula(formula.call) , data=vocSelTable,
                            subset = (Features == featureAF & ( (CallTest != 'Be') & (CallTest != 'LT'))), family= binomial)
  (summary(model.CallerVoice))

  model.Null <- glmer( as.formula(formula.1) , data=vocSelTable, 
                    subset = (Features == featureAF & ( (CallTest != 'Be') & (CallTest != 'LT'))) , family= binomial)
  anova(model.Null, model.CallerVoice, test = 'Chisq')


  model.CallerVoice.Effect <- effect("CallTest:CallTrain", model.CallerVoice, se=TRUE, confidence.level = 0.99)
  sum.CallerVoice.Effect <- summary(model.CallerVoice.Effect)

  # Not sure if this is useful 
  model.CallerVoice.Table <- cbind(Effect=sum.CallerVoice.Effect$effect, LB=sum.CallerVoice.Effect$lower, UB=sum.CallerVoice.Effect$upper)

  plot(model.CallerVoice)
  plot(model.CallerVoice.Effect)

  write.csv(sum.CallerVoice.Effect$effect, file = fileModelCoefMean[[ic]])
  write.csv(sum.CallerVoice.Effect$lower, file = fileModelCoefLB[[ic]])
  write.csv(sum.CallerVoice.Effect$upper, file = fileModelCoefUB[[ic]])

}
