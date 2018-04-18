# caret-workshop
Applied Predictive Modeling with ```caret```  

## Basics
Create synthetic data using ```twoClassSim```
Quickly explore the data using ```skimr``` and ```xray```  
Split the dataset into train/test with an index  

## TrainControl
Setting up train control in ```caret```  
Cross-validation method and settings  
Subsampling to deal with class-imbalance (mentioned but not implemented)  

## ModelTrain
Placeholder regression example  
Classification example  
Logistic Regression (```glm```), Elastic Net (```glmnet```), Random Forest (```ranger```)  
Using summary, variable importance, plot on fit object  
Prediction on unseen data: class; class probability  

## Performance Metrics
In-sample: ROC, Sensitivity (true positive rate), Specificity (true negative rate)  
Confusion matrices

## Ensembles
Model dissimilarity using Jaccard distance  
Linear ensembles  
Meta-Model ensembles  

## Feature Selection
Recursive Feature Elimination  
Simulated Annealing  
Genetic Algorithm  
