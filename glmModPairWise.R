# Calculates mixed effect glm on Voice.
library(lme4)
library(effects)
library(car)

# Read Excel data file.
fileCSVTable = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerDiscrimResults.csv'
fileModelCoef = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerDiscrimGLMModel.csv'
  
vocSelTable <- read.csv(fileCSVTable, header = TRUE)

ndata <- sum(vocSelTable$Features == '18 AF')

# This number must correspond to the number of cross validations

model.Pair.Voice <- glmer( cbind(LDAYes, Count-LDAYes) ~ Type + (1| BirdPair) , data=vocSelTable, subset = Features == '18 AF', family= binomial)
model.Null <- glmer( cbind(LDAYes, Count-LDAYes) ~ 1 + (1| BirdPair) , data=vocSelTable, subset = Features == '18 AF', family= binomial)

anova(model.Null, model.Pair.Voice, test = 'Chisq')

model.Effect <- effect("Type", model.Pair.Voice, se=TRUE)
sum.Effect <- summary(model.Effect)

model.Table <- cbind(Effect=sum.Effect$effect, LB=sum.Effect$lower, UB=sum.Effect$upper)

write.csv(model.Table, file = fileModelCoef)
