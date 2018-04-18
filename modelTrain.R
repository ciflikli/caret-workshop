#install.packages(c("caret", "caretEnsemble", "skimr", "xray", "proxy", "doParallel"))

library(caret)
library(caretEnsemble)
library(skimr)
library(xray)
library(proxy)
library(doParallel)
registerDoParallel(detectCores() - 1)

#load("caret.rdata")

####Basics####
dat <- twoClassSim(n = 1000, linearVars = 2, noiseVars = 5, corrVars = 2, mislabel = .01)
skim(dat)
anomalies(dat)

####Train/Test Split####
index <- createDataPartition(y = dat$Class, p = .7, list = FALSE)
training <- dat[index, ]
test <- dat[-index, ]

####trainControl####
reg.ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5, allowParallel = TRUE)

cls.ctrl <- trainControl(method = "repeatedcv", #boot, cv, LOOCV, timeslice OR adaptive etc.
                         number = 10, repeats = 5,
                         classProbs = TRUE, summaryFunction = twoClassSummary,
                         savePredictions = "final", allowParallel = TRUE)

####Regression####

set.seed(1895)
lm.fit <- train(Linear1 ~ ., data = training, trControl = reg.ctrl, method = "lm")
lm.fit

####Classification####
set.seed(1895)
glm.fit <- train(Class ~ ., data = training, trControl = cls.ctrl,
                 method = "glm", family = "binomial", metric = "ROC",
                 preProcess = c("nzv", "center", "scale"))
glm.fit

##For reference - alternate formula interface
y <- training$Class
predictors <- training[,which(colnames(training) != "Class")]

set.seed(1895)
glm.fit <- train(x = predictors, y = y, trControl = cls.ctrl,
                 method = "glm", family = "binomial", metric = "ROC",
                 preProcess = c("nzv", "center", "scale"))

glm.fit
plot(varImp(glm.fit))

##Predict on unseen data
glm.preds <- predict(glm.fit, newdata = test)
head(glm.preds, 20)
glm.prob.preds <- predict(glm.fit, newdata = test, type = "prob") #class probabilities
head(glm.prob.preds)

##Elastic Net
glmnet.fit <- train(x = predictors, y = y, trControl = cls.ctrl,
                    method = "glmnet", metric = "ROC",
                    preProcess = c("nzv", "center", "scale"),
                    tuneGrid = expand.grid(alpha = 0:1,
                                           lambda = seq(0.0001, 1, length = 20)))
glmnet.fit
plot(glmnet.fit)

##Random Forest

set.seed(1895)
rf.fit <- train(Class ~ ., data = training, trControl = cls.ctrl,
                method = "ranger", metric = "ROC",
                preProcess = c("nzv", "center", "scale"))
rf.fit
plot(rf.fit)

rf.preds <- predict(rf.fit, newdata = test, type = "prob")
head(rf.preds)

####Multiple Models####
set.seed(1895)
models <- caretList(Class ~ ., data = training, trControl = cls.ctrl, metric = "ROC",
                    tuneList = list(logit = caretModelSpec(method = "glm", family = "binomial"),
                                    elasticnet = caretModelSpec(method = "glmnet", tuneGrid = expand.grid(alpha = 0:1, lambda = seq(0.0001, 1, length = 20))),
                                    rf = caretModelSpec(method = "ranger")),
                    preProcess = c("nzv", "center", "scale"))
models

models.preds <- lapply(models, predict, newdata = test)
models.preds <- data.frame(models.preds)
head(models.preds, 10)

####Model Dissimilarity####
tag <- read.csv("tag_data.csv", row.names = 1)
tag <- as.matrix(tag)

classModels <- tag[tag[, "Classification"] == 1,]

all <- 1:nrow(classModels)
start <- grep("ranger", rownames(classModels), fixed = TRUE)
pool <- all[all != start]

nextMods <- maxDissim(classModels[start,,drop = FALSE], 
                      classModels[pool, ], 
                      method = "Jaccard",
                      n = 4)

rownames(classModels)[c(start, nextMods)]

####Performance Metrics####

##Single Model Fit
confusionMatrix(rf.fit)

##Multiple Model Fits
bwplot(resamples(models)) #dotplot

####Ensembles####

##Linear (Simple) Ensembles
xyplot(resamples(models))

set.seed(1895)
greedy_ensemble <- caretEnsemble(models, metric = "ROC", trControl = cls.ctrl)
summary(greedy_ensemble)

##Meta-Model Ensembles
glm_ensemble <- caretStack(models, method = "glm", metric = "ROC", trControl = cls.ctrl)
summary(glm_ensemble)

####Feature Selection####

####Recursive Feature Elimination

lrFuncs$summary <- twoClassSummary
rfe.ctrl = rfeControl(functions = lrFuncs,
                      method = "boot",
                      number = 25,
                      allowParallel = TRUE, verbose = TRUE)

subsets <- c(1:length(training))

set.seed(1895)
rfe <- rfe(x = predictors, y = y, sizes = subsets,
           metric = "ROC", rfeControl = rfe.ctrl)
rfe

####Simulated Annealing

caretSA$fitness_extern <- twoClassSummary

safs.ctrl = safsControl(functions = caretSA, method = "boot", number = 10,
                        metric = c(internal = "ROC", external = "ROC"),
                        maximize = c(internal = TRUE, external = TRUE),
                        holdout = .2, improve = 5,
                        allowParallel = TRUE, verbose = TRUE)

sa <- safs(x = training[,which(colnames(training) != "Class")],
           y = training$Class,
           iters = 10, method = "glm", family = "binomial", metric = "ROC",
           trControl = cls.ctrl,
           safsControl = safs.ctrl)
sa

####Genetic Algorithm
caretGA$fitness_extern <- twoClassSummary

gafs.ctrl = gafsControl(functions = caretGA, method = "boot", number = 10,
                        metric = c(internal = "ROC", external = "ROC"),
                        maximize = c(internal = TRUE, external = TRUE),
                        holdout = .2,
                        allowParallel = TRUE, genParallel = TRUE, verbose = TRUE)

set.seed(1895)
ga <- gafs(x = predictors, y = y, iters = 5, popSize = 2, elite = 0,
           differences = TRUE, method = "glm", family = "binomial", metric = "ROC",
           trControl = cls.ctrl,
           gafsControl = gafs.ctrl)
ga