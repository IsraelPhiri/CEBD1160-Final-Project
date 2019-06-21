# CEBD1160_Final_Project

| Name         | Date           |
|:-------------|:---------------|
|Israel Phiri  | June 21, 2019. |

-----

### Resources
My repository includes the following:

- Python script for analysis: project-processor.py
- Results figure/saved file : alcohol_distribution.png multi_pairplot.png distplots(dir) matplots(dir) pairplots(dir)
- Dockerfile for your experiment: Dockerfile  requirements.txt
- runtime-instructions          : RUNME.md
- Explanation of my project : README.md
- Extras: wine.data

-----

## Research Question

1 Wine has features that can be identified and their amounts quantified in a sample. These features do influence the classification of wine. With this wine data set, could we train a model to predict wine sample classification? 

### Abstract

- The wine recognition data set categorizes wine according to Class 1, 2 or 3.  
- From multi-pair plot alcohol, flavonoids, proline and total phenols features seem to influence  
  classification the most. 
- The distribution plot reveals that the three wine classes are indeed separable, in the plot 
  given they are separated by alcohol content. 
- I then tried at least two models, ie KNN and LogisticRegression models, to see how each performed in predicting
  the classification of unknown wine samples. 
- The logistic regression predictive model was the best of the two with a 94% f1-score. I tried to improve the prediction of
  the model by using confusion matrix, to no avail.
- Scaling or standardizing the data before training the model could have improved the perfomance of the model. 

### Introduction

The wine dataset is the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars.The analysis determined the quantities of 13 constituents found in each of the three types of wines.(1)
The 13 features that were analyzed are Alcohol, Malic acid, Ash, Alcalinity of ash, Magnesium, Total phenols, Flavanoids
Nonflavanoid phenols, Proanthocyanins, Color Intensity, Hue, OD280/OD315 of diluted wines and Proline. Depending on the influencies of each of these features, the wine can be classified as class 0, 1 or 2.

### Methods

- To solve my question, first I used the distribution plot so as to have an idea if the targets could be
  separated.(Please see alcohol.png plot)
- I then split the data(0.35) for training and testing.
- I tried at least two predictive models ( could have tried more ) and picked the best between the two based on their
  prediction results.
- The logistic regression was the better perfomer with an f1-score of 94%.  

### Results

- figures and plots ( please see distplots/ matplots/ pair_plots/ and multi_pairplot.png


LogisticRegression Accuracy score is 0.95/ 

LogisticRegression Classification Report
precision    recall  f1-score   support

0       0.95      1.00      0.97        18
1       1.00      0.90      0.95        29
2       0.89      1.00      0.94        16

accuracy                           0.95        63
macro avg       0.95      0.97      0.95        63
weighted avg       0.96      0.95      0.95        63

LogisticRegression Confusion Matrix
[[18  0  0]
[ 1 26  2]
[ 0  0 16]]
LogisticRegression Overall f1-score
0.9532013296719178
LogisticRegression Cross validation score:[0.94594595 0.97222222 0.91666667 1.         1.        ]

LogisticRegression ShuffleSplit val_score:[1.         1.         0.94444444 1.         1.        ]



KNeighbours Accuracy score is 0.67/ 

KNeighbours Classification Report
precision    recall  f1-score   support

0       0.70      0.89      0.78        18
1       0.78      0.62      0.69        29
2       0.47      0.50      0.48        16

accuracy                           0.67        63
macro avg       0.65      0.67      0.65        63
weighted avg       0.68      0.67      0.66        63

KNeighbours Confusion Matrix
[[16  0  2]
[ 4 18  7]
[ 3  5  8]]
KNeighbours Overall f1-score
0.6525479940114086
KNeighbours Cross validation score:[0.62162162 0.72222222 0.66666667 0.65714286 0.85294118]

KNeighbours ShuffleSplit val_score:[0.72222222 0.83333333 0.66666667 0.72222222 0.66666667]

- A short explanation of both of the above: The results above show that for this particular dataset and between these two methods, the Logistic Regression model predicts much better than the KNeighbours model. It can be see than from the start, the accuracy of the LR model is 0.95 compared to 0.67 for KNN. A confusion metrix is applied and so is the ShuffleSplit, to try and improve each model's performance but still the LR model predicts better.

### Discussion
What I did:
The data and target are split 0.65 to 0.35 between trainer and tester. Each method is trained on the same execution then tested to see their individual predictions. First, the accuracy of each model is analyzed, then Confusion Matrix and ShuffleSplit are applied to try and improve each model's performance. Both results have the LR out performing the KNN in predicting the wine classification for this dataset, hense that's why it is chosen. 
With an f1-score of 94%, I think the training and choice of my model was fairly acceptable, though there is room for
improvement.
The next improvement to make would be  to standardize/normalize the data before training the predictive model. This would rescale the data within a range of 0 and 1 to reduce data noise and give a clearer perspective.

### References
All of the links
Citation:
(1) Lichman, M. (2013). UCI Machine Learning Repository [https://archive.ics.uci.edu/ml].
 Irvine, CA: University of California, School of Information and Computer Science.

-------
