library(caret)
library(caretEnsemble)
library(skimr)
library(doParallel)
registerDoParallel(detectCores() - 1)

####Basics####
dat <- twoClassSim(n = 1000, linearVars = 2, noiseVars = 5, corrVars = 2, mislabel = .01)
skim(dat)
xray::anomalies(dat)

####Train/Test Split####
index <- createDataPartition(y = dat$Class, p = .7, list = FALSE)
training <- dat[index, ]
test <- dat[-index, ]

####trainControl####
reg_ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5, allowParallel = TRUE)

cls_ctrl <- trainControl(method = "repeatedcv", #boot, cv, loocv, timeslice etc.
                         number = 10, repeats = 5,
                         classProbs = TRUE, summaryFunction = twoClassSummary,
                         sampling = "up", #down, SMOTE, ROSE
                        savePredictions = "final", allowParallel = TRUE)

####Regression####

####Classification####
set.seed(1895)
glm.fit <- train(Class ~ ., data = training, trControl = cls_ctrl, method = "glm", family = "binomial",
                 metric = "ROC")
glm.fit
plot(varImp(glm.fit))

glm.preds <- predict(glm.fit, newdata = test, type = "prob")

set.seed(1895)
rf.fit <- train(Class ~ ., data = training, trControl = cls_ctrl, method = "ranger",
                metric = "ROC")
rf.fit
plot(rf.fit)

rf.preds <- predict(rf.fit, newdata = test, type = "prob")

####Multiple Models####
set.seed(1895)
models <- caretList(Class ~ ., data = training, trControl = cls_ctrl, metric = "ROC",
                    tuneList = list(logit = caretModelSpec(method = "glm", family = "binomial"),
                                    rf = caretModelSpec(method = "ranger")))
models.preds <- lapply(models, predict, newdata = test, type = "prob")
models.preds <- data.frame(models.preds)

####Performance Metrics####
resamples(models)
bwplot(resamples(models)) #dotplot

####Ensembles####

##Linear (Simple) Ensembles
xyplot(resamples(models))
set.seed(1895)
greedy_ensemble <- caretEnsemble(models, metric = "ROC", trControl = cls_ctrl)
summary(greedy_ensemble)

##Meta-Model Ensembles
glm_ensemble <- caretStack(models, method = "glm", metric = "ROC", trControl = cls_ctrl)
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
rfe <- rfe(x = training[,which(colnames(training) != "Class")],
           y = training$Class, sizes = subsets, metric = "ROC", rfeControl = rfe.ctrl)
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
           trControl = cls_ctrl,
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
ga <- gafs(x = training[,which(colnames(training) != "Class")],
           y = training$Class, iters = 5, popSize = 2, elite = 0,
           differences = TRUE, method = "glm", family = "binomial", metric = "ROC",
           trControl = cls_ctrl,
           gafsControl = gafs.ctrl)
ga