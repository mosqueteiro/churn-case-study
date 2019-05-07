
The Gradient Boosting Algorithm Performed slightly better than our Random Forest.

The default valued version resulted in:

> confusion Matrix:   
 [[1987  705]  
 [ 985 4323]]  

> Accuracy:  0.789  
> Precision:  0.738  
> Recall:  0.669  
> AUC:  0.764  

Notable default values:
* N_Estimators = 100
* Learning Rate = 0.1

![alt text](ImagesDwl/model1.png)

To optimize, we iterated through various collections of:
* N_Estimators, Ideal = 220
* Learning Rate, Ideal = 0.2

> Confusion Matrix:    
[[1995  684]
 [ 977 4344]]
> Accuracy:  0.792  
Precision:  0.745  
Recall:  0.671  
AUC:  0.768

![alt text](ImagesDwl/model2.png)

Changes between models:

> Accuracy Delta:  0.00362  
Precision Delta:  0.00656  
Recall Delta:  0.00269  
AUC Delta:  0.00343  

Feature Importances   

![alt text](ImagesDwl/modelFeature.png)

Notes:

From this, we infer that the driver rating of the passenger was a highly important feature to include.
Further, city (categorical) and phone OS (categorical) were also important but we cannot act on magnitude or direction like with a coefficient.

Within SKLearn:
* To determine quality of split, friedman_mse is utilized as the default metric. 
* Fit on negative gradient of previous split, minimizing deviance
