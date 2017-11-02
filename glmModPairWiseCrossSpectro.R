# Calculates mixed effect glm on Voice.
library(lme4)
library(effects)
library(car)

# Read Excel data file.
fileCSVTable = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossDiscrimSpectroResults.csv'
vocSelTable <- read.csv(fileCSVTable, header = TRUE)


# Set up the output files
fileModelCoefLDA = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossSpectroGLMModelLDA.csv'
fileModelCoefQDA = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossSpectroGLMModelQDA.csv'
fileModelCoefRF =  '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/pairCallerCrossSpectroGLMModelRF.csv'


fileModel = as.list(c(fileModelCoefLDA, fileModelCoefQDA, fileModelCoefRF))
classifier = as.list(c('LDA', 'QDA', 'RF'))

for (ic in 1:length(classifier)) {
  ndata <- sum(vocSelTable$Features == 'Spectro')
  print(sprintf('N=%d for Spectro', ndata))

# First the model for the voice only: from training on other categories and fitting on one
# Note that this data only has adult calls because 
  formula.call <- sprintf('cbind(%sYes, Count-%sYes) ~ Call + (1| BirdPair)', classifier[[ic]], classifier[[ic]])
  formula.1 <- sprintf('cbind(%sYes, Count-%sYes) ~ 1 + (1| BirdPair)', classifier[[ic]], classifier[[ic]])
  model.Voice <- glmer( as.formula(formula.call) , data=vocSelTable, subset = (Features == 'Spectro' & TestType =='Voice'), family= binomial)
  model.Null <- glmer( as.formula(formula.1) , data=vocSelTable, subset = (Features == 'Spectro' & TestType =='Voice') , family= binomial)
  anova(model.Null, model.Voice, test = 'Chisq')

  model.Voice.Effect <- effect("Call", model.Voice, se=TRUE)
  sum.Voice.Effect <- summary(model.Voice.Effect)
  model.Voice.Table <- cbind(Effect=sum.Voice.Effect$effect, LB=sum.Voice.Effect$lower, UB=sum.Voice.Effect$upper)

  formula.all <- sprintf('cbind(%sYes, Count-%sYes) ~ Call*TestType + (1| BirdPair)',  classifier[[ic]], classifier[[ic]])
  model.CallerVoice <- glmer( as.formula(formula.all) , data=vocSelTable, subset = (Features == 'Spectro'), family= binomial)
  (summary(model.CallerVoice))

  model.CallerVoice.Effect <- effect("Call:TestType", model.CallerVoice, se=TRUE)
  sum.CallerVoice.Effect <- summary(model.CallerVoice.Effect)
  model.CallerVoice.Table <- cbind(Effect=sum.CallerVoice.Effect$effect, LB=sum.CallerVoice.Effect$lower, UB=sum.CallerVoice.Effect$upper)

  plot(model.CallerVoice)
  plot(model.CallerVoice.Effect)

  write.csv(model.CallerVoice.Table, file = fileModel[[ic]])

}
