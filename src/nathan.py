from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold
import sklearn.metrics as metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.case_study_funcs import clean
from src.case_study_funcs import plot_roc


# pull in data
df = pd.read_csv('data/churn_train.csv')


# clean up data
drop = ['surge_pct']
df_ = clean(df, drop)

# Train test split
y = df_['churn'].values
X = df_.drop(['churn'], axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                test_size=.2, random_state=42)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=1, oob_score=True,
                            bootstrap=True)
fig, ax = plt.subplots(figsize=(10,8))
plot_roc(X, y, rf, fig, title='RandomForestClassifier')

fig.show()
fig.savefig('images/rf_roc.png')

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=1,
                                learning_rate = 0.1)
fig, ax = plt.subplots(figsize=(10,8))
plot_roc(X, y, rf, fig, title=gb.__class__.__name__)

fig.show()
fig.savefig('images/gb_roc.png')
