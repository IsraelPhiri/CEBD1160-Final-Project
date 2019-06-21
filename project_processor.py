from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import os
import numpy as np

# load the wine dataset from sklearn datasets
wine_raw_data = datasets.load_wine()

#transform the wine_raw_data into a DataFrame with all the features as columns
features = pd.DataFrame(data = wine_raw_data['data'],columns=wine_raw_data['feature_names'])
wine_df = features


# adding 'target' column to wine_df DataFrame and filling it with data from wine_raw_data 'target'
wine_df['target']=wine_raw_data['target']

#assign viriables to data and target
X = wine_raw_data.data
y = wine_raw_data.target

# Splitting datasets into: train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.35)

# Training a Linear Regression model with fit()
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=7000)
lr.fit(X_train, y_train)

#Training a KNeighbours Classifier model with fit()
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3,weights='uniform',algorithm='auto')
kn.fit(X_train,y_train)

# Output of the LR training is a model: a + b*X0 + c*X1 + d*X2 ...
print(f"LogisticRegression Intercept per class: {lr.intercept_}\n")
print(f"LogisticRegression Coeficients per class: {lr.coef_}\n")

print(f"LogisticRegression Available classes: {lr.classes_}\n")

print(f"LogisticRegression Number of iterations generating model: {lr.n_iter_}\n")

# Predicting the LR results for test dataset
predicted_values = lr.predict(X_test)

# Printing the LR residuals: difference between real and predicted
for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f'Value: {real}, LogisticRegression pred: {predicted} {"Ouch!!" if real != predicted else ""}\n')


# Printing LR accuracy score(mean accuracy) from 0 - 1
print(f'LogisticRegression Accuracy score is {lr.score(X_test, y_test):.2f}/ \n')

# Printing the LR classification report
from sklearn.metrics import classification_report, confusion_matrix, f1_score
print('LogisticRegression Classification Report')
print(classification_report(y_test, predicted_values))

# Printing the LR classification confusion matrix (diagonal is true)
print('LogisticRegression Confusion Matrix')
print(confusion_matrix(y_test, predicted_values))

print('LogisticRegression Overall f1-score')
print(f1_score(y_test, predicted_values, average="macro"))

# Cross validation LR using cross_val_score
from sklearn.model_selection import cross_val_score, ShuffleSplit
print(f'LogisticRegression Cross validation score:{cross_val_score(lr, X, y, cv=5)}\n')

# Cross validation LR using shuffle split
cv = ShuffleSplit(n_splits=5)
print(f'LogisticRegression ShuffleSplit val_score:{cross_val_score(lr, X, y, cv=cv)}\n')
print()



# Predicting the results for test dataset using KNeighbours
kn_predicted_values = kn.predict(X_test)

# Printing the residuals: difference between real and KN predicted
for (real, predicted) in list(zip(y_test, kn_predicted_values)):
    print(f'Value: {real}, KNeighbours pred: {predicted} {"Ouch!!" if real != predicted else ""}\n')

# Printing KN accuracy score(mean accuracy) from 0 - 1
print(f'KNeighbours Accuracy score is {kn.score(X_test, y_test):.2f}/ \n')

# Printing the KN classification report
from sklearn.metrics import classification_report, confusion_matrix, f1_score
print('KNeighbours Classification Report')
print(classification_report(y_test, kn_predicted_values))

# Printing the KN classification confusion matrix (diagonal is true)
print('KNeighbours Confusion Matrix')
print(confusion_matrix(y_test, kn_predicted_values))

print('KNeighbours Overall f1-score')
print(f1_score(y_test, kn_predicted_values, average="macro"))

# Cross validation KN using cross_val_score
from sklearn.model_selection import cross_val_score, ShuffleSplit
print(f'KNeighbours Cross validation score:{cross_val_score(kn, X, y, cv=5)}\n')

# Cross validation KN using shuffle split
cv = ShuffleSplit(n_splits=5)
print(f'KNeighbours ShuffleSplit val_score:{cross_val_score(kn, X, y, cv=cv)}\n')

# create directory for plots
os.makedirs('plots/project1160/distplots', exist_ok=True)
os.makedirs('plots/project1160/matplots', exist_ok=True)
os.makedirs('plots/project1160/pair_plots', exist_ok=True)


for feature in wine_raw_data['feature_names']:
    gs1 = gridspec.GridSpec(3,1)
    ax1 = plt.subplot(gs1[:-1])
    ax1.set_title('{}'.format(feature)+' content per class')
    gs1.update(right=0.9)
    sns.kdeplot(wine_df[feature][wine_df.target==0],ax=ax1,label='0')
    sns.kdeplot(wine_df[feature][wine_df.target==1],ax=ax1,label='1')
    sns.kdeplot(wine_df[feature][wine_df.target==2],ax=ax1,label='2')
    ax1.xaxis.set_visible(False)
#because file name can not have a '/' , had to put '-' in od280/od315_of_diluted_wines file name
    if feature!= 'od280/od315_of_diluted_wines':
        plt.savefig(f'plots/project1160/matplots/'+str(feature)+'_content.png')
    else:
        plt.savefig(f'plots/project1160/matplots/od280-od315_of_diluted_wines_content.png')
plt.clf()

# visualization of each feature distribution on the three targets/classes

for each_target in wine_df.target.unique():
    sns.distplot(wine_df['alcohol'][wine_df.target==each_target], kde = 1, label ='{}'.format(each_target))
    plt.legend()
    plt.savefig(f'plots/project1160/distplots/alcohol.png', dpi=300)
plt.clf()

#Wine Pairplot with hue and 'diag_kind=histogram'
sns.pairplot(wine_df,hue = 'target', diag_kind='hist')
plt.savefig(f'plots/project1160/pair_plots/pairplot.png')
plt.clf()

