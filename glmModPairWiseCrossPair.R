# Calculates mixed effect glm on Voice.
library(lme4)
library(effects)
library(car)

# Run on this model
featureAF = 'Fund AF'

# Read Excel data file.
fileCSVTable = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossPairDiscrimResults.csv'
vocSelTable <- read.csv(fileCSVTable, header = TRUE)

# Output files
fileModelCoefMean = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossPairGLMModelMeanFund.csv'
fileModelCoefLB = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossPairGLMModelLBFund.csv'
fileModelCoefUB = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossPairGLMModelUBFund.csv'

vocSelTable <- read.csv(fileCSVTable, header = TRUE)
ndata <- sum(vocSelTable$Features == featureAF)

# Separate between Adults and Juveniles

model.CallerVoice <- glmer( cbind(LDAYes, Count-LDAYes) ~ CallTest*CallTrain + (1| BirdPair) , data=vocSelTable,
                            subset = (Features == featureAF & ( (CallTest != 'Be') & (CallTest != 'LT'))), family= binomial)
(summary(model.CallerVoice))

model.Null <- glmer( cbind(LDAYes, Count-LDAYes) ~ 1 + (1| BirdPair) , data=vocSelTable, 
                    subset = (Features == featureAF & ( (CallTest != 'Be') & (CallTest != 'LT'))) , family= binomial)
anova(model.Null, model.CallerVoice, test = 'Chisq')


model.CallerVoice.Effect <- effect("CallTest:CallTrain", model.CallerVoice, se=TRUE, confidence.level = 0.99)
sum.CallerVoice.Effect <- summary(model.CallerVoice.Effect)

# Not sure if this is useful 
model.CallerVoice.Table <- cbind(Effect=sum.CallerVoice.Effect$effect, LB=sum.CallerVoice.Effect$lower, UB=sum.CallerVoice.Effect$upper)

plot(model.CallerVoice)
plot(model.CallerVoice.Effect)

write.csv(sum.CallerVoice.Effect$effect, file = fileModelCoefMean)
write.csv(sum.CallerVoice.Effect$lower, file = fileModelCoefLB)
write.csv(sum.CallerVoice.Effect$upper, file = fileModelCoefUB)
