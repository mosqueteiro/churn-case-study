import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import clean as clean
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

def confusionMatrix(estimator, y_test, y_pred):
    acc = np.mean(y_test == y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    matrix = np.array([[tp,fp],[fn, tn]])
    pres = tp / (tp + fp)
    recall = tp / (tp + fn)
    oob_score = estimator.oob_score_
    return acc, matrix, pres, recall, oob_score

if __name__ == "__main__":
    #importing the data
    df_train_imp = pd.read_csv('data/churn_train.csv')
    df_test_imp = pd.read_csv('data/churn_test.csv')

    #cleaning the data
    drop = ['surge_pct']
    df_train = clean.clean(df_train_imp, drop)
    df_test = clean.clean(df_test_imp, drop)
    
    #spliting X and y for training and test
    y_train = df_train['churn'].values
    X_train = df_train.drop(['churn'], axis=1).values
    y_test = df_test['churn'].values
    X_test = df_test.drop(['churn'], axis=1).values
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_train, y_train, test_size=.2, random_state=42)
    
    #build random forest
    rf = RandomForestClassifier(n_estimators=100, random_state=1, oob_score=True, bootstrap=True)
    rf.fit(X_train_s, y_train_s)
    y_pred = rf.predict(X_test_s)

    plt.bar(df_train.drop("churn", axis=1).columns, rf.feature_importances_)
    plt.ylabel('feature importance', fontsize=14)
    plt.show()

    #confusionMatrix
    acc, matrix, pres, recall, oob_score = confusionMatrix(rf, y_test_s, y_pred)
    print('acc: ', acc)
    print('confusion Matrix: ', matrix)
    print('pres: ', pres)
    print('recall: ', recall)
    print('oob: ', oob_score)

    sns.heatmap(matrix, annot=True)
    plt.show()

    # num_tree = [1,2,3,4,5,6,7,8,9]
    # randstate = [1,2,3]
    # for j in randstate:
    #     acc = []
    #     for i in num_tree:
    #         rf = RandomForestClassifier(n_estimators=100,random_state=j,max_features=i)
    #         rf.fit(X_train, y_train)
    #         y_pred = rf.predict(X_test)
    #         acc.append(np.mean(y_test == y_pred))
    #     plt.plot(num_tree, acc)
    # plt.xlabel('features')
    # plt.ylabel('accuracy')
    # plt.show()

    y_pred = rf.predict(X_test)

    #confusionMatrix
    acc, matrix, pres, recall, oob_score = confusionMatrix(rf, y_test, y_pred)
    print('acc: ', acc)
    print('confusion Matrix: ', matrix)
    print('pres: ', pres)
    print('recall: ', recall)
    print('oob: ', oob_score)

    sns.heatmap(matrix, annot=True)
    plt.show()