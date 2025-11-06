library(devtools); install_github('JeffreyRacine/npiv')

library("npiv")
data("Engel95", package = "npiv")
Engel95 <- Engel95[order(Engel95$logexp),] 
attach(Engel95)
logexp.eval <- seq(4.5,6.5,length=100)

names(Engel95)

hist(Engel95$logwages)
hist(Engel95$logexp.eval)

food_engel <- npiv(food, logexp, logwages, X.eval = logexp.eval)
summary(food_engel)
plot(food_engel, showdata = TRUE)