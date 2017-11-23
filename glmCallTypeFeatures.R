# Calculates mixed effect glm on Call type Data
# This is R version of a script that does now work in Python.
library(lme4)
library(effects)
library(car)

Yes <- c(92, 114, 79)
Count <- c(182, 182, 182)
Feature <- c('Fund', 'Spec', 'Temp')

dataLDACall <- data.frame(Yes, Count, Feature)

model.glm <- glm(cbind(Yes, Count-Yes) ~ Feature, data=dataLDACall, family=binomial)
model.null <- glm(cbind(Yes, Count-Yes) ~ 1, data=dataLDACall, family=binomial) 
summary(model.glm)
anova(model.null, model.glm, test='Chisq')
