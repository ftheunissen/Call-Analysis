# Calculates mixed effect glm on Voice.
library(lme4)
library(effects)
library(car)

# Read Excel data file.
fileCSVTable = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerDiscrimResults.csv'
fileModelCoef = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerDiscrimGLMModel.csv'
fileModelSexCoef = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerSexDiscrimGLMModel.csv'
fileModelJuvSexCoef = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerJuvSexDiscrimGLMModel.csv'

vocSelTable <- read.csv(fileCSVTable, header = TRUE)

ndata <- sum(vocSelTable$Features == '18 AF')

# This number must correspond to the number of cross validations

model.Voice <- glmer( cbind(LDAYes, Count-LDAYes) ~ Type + (1| BirdPair) , data=vocSelTable, subset = Features == '18 AF', family= binomial)
model.Null <- glmer( cbind(LDAYes, Count-LDAYes) ~ 1 + (1| BirdPair) , data=vocSelTable, subset = Features == '18 AF', family= binomial)

anova(model.Null, model.Voice, test = 'Chisq')

model.Effect <- effect("Type", model.Voice, se=TRUE)
sum.Effect <- summary(model.Effect)

model.Table <- cbind(Effect=sum.Effect$effect, LB=sum.Effect$lower, UB=sum.Effect$upper)

write.csv(model.Table, file = fileModelCoef)

# Models for Type and Sex in Adult Calls - Whines
model.AdultPair.Voice <- glmer( cbind(LDAYes, Count-LDAYes) ~ Type + SexPair + Type:SexPair + (1| BirdPair) , data=vocSelTable, 
                                subset = (Features == '18 AF') & (SexPair != 'U') & (Type != 'Caller So') & (Type != 'Caller Be') & (Type != 'Caller LT') & (Type != 'Caller Wh'),
                                family= binomial)

model.Adult.Voice <- glmer( cbind(LDAYes, Count-LDAYes) ~ Type + (1| BirdPair) , data=vocSelTable, 
                            subset = (Features == '18 AF') & (SexPair != 'U') & (Type != 'Caller So') & (Type != 'Caller Be') & (Type != 'Caller LT') & (Type != 'Caller Wh'), 
                            family= binomial)

summary(model.AdultPair.Voice)
summary(model.Adult.Voice)
anova(model.Adult.Voice, model.AdultPair.Voice, test = 'Chisq')

model.Pair.Effect <- effect("Type:SexPair", model.AdultPair.Voice, se=TRUE)
plot(model.Pair.Effect)
sum.Effect <- summary(model.Pair.Effect)

model.AdultPairTable <- cbind(Effect=sum.Effect$effect, LB=sum.Effect$lower, UB=sum.Effect$upper)

write.csv(model.AdultPairTable, file = fileModelSexCoef)

# Models for Type and Sex in Juvenile Calls
model.JuvPair.Voice <- glmer( cbind(LDAYes, Count-LDAYes) ~ Type + SexPair + Type:SexPair + (1| BirdPair) , data=vocSelTable, 
                                subset = (Features == '18 AF') & (SexPair != 'U') & ((Type == 'Caller Be') | (Type == 'Caller LT')),
                                family= binomial)

model.Juv.Voice <- glmer( cbind(LDAYes, Count-LDAYes) ~ Type + (1| BirdPair) , data=vocSelTable, 
                            subset = (Features == '18 AF') & (SexPair != 'U') & ((Type == 'Caller Be') | (Type == 'Caller LT')),
                            family= binomial)

summary(model.JuvPair.Voice)
summary(model.Juv.Voice)
anova(model.Juv.Voice, model.JuvPair.Voice, test = 'Chisq')

model.JuvPair.Effect <- effect("Type:SexPair", model.JuvPair.Voice, se=TRUE)
plot(model.JuvPair.Effect)
sum.JuvEffect <- summary(model.JuvPair.Effect)

model.JuvPairTable <- cbind(Effect=sum.JuvEffect$effect, LB=sum.JuvEffect$lower, UB=sum.JuvEffect$upper)

write.csv(model.JuvPairTable, file = fileModelJuvSexCoef)

