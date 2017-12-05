# Author:   Jay Huang
# E-mail:   askjayhuang at gmail dot com
# GitHub:   https://github.com/jayh1285
# Created:  2017-11-21T19:02:57.726Z

"""A module for predicting movement in the Dow Jones based off the top 25 news
   headlines of the day using natural language processing and binary
   classification concepts.
"""

################################################################################
# Imports
################################################################################

import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

################################################################################
# Global Variables
################################################################################

CSV_PATH = 'data/Combined_News_DJIA.csv'

################################################################################
# Functions
################################################################################


def combine_text_columns(df):
    """Combine all text columns in a row of a DataFrame."""
    # Fill non-null values to be an empty string
    df.fillna("", inplace=True)

    # Join all text columns in a row with a space in between
    df = df.apply(lambda x: " ".join(x), axis=1)

    return df

################################################################################
# Execution
################################################################################


if __name__ == '__main__':
    warnings.simplefilter(action='ignore')

    # Read data from csv file and clean DataFrame
    df = pd.read_csv(CSV_PATH, index_col=0)
    df.index = pd.DatetimeIndex(df.index)

    # Set X and y
    X = df.drop('Label', axis=1)
    y = df['Label']

    # Split data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Show tokenized words for the first row
    X_combined = combine_text_columns(X)
    tokenizer = CountVectorizer().build_tokenizer()(X_combined.iloc[0])
    df = pd.DataFrame([[x, tokenizer.count(x)] for x in set(tokenizer)], columns=['Word', 'Count'])
    df.sort_values('Count', inplace=True, ascending=False)
    print(X.iloc[0].name, '\n')
    print(X_combined.iloc[0], '\n')
    print(df.head(15), '\n')

    # Create a FunctionTransfomer to combine text columns in a row
    combine_text_ft = FunctionTransformer(combine_text_columns, validate=False)

    # Create pipeline
    pl = Pipeline([
        ('cmb', combine_text_ft),
        ('vct', CountVectorizer(ngram_range=(2, 2))),
        ('int', SparseInteractions(degree=2)),
        ('clf', LogisticRegression(C=.027, solver='sag'))
    ])

    # # Grid search cross validation for C in LogisticRegression
    # Cs = np.logspace(-2, 2, 10)
    #
    # estimator = GridSearchCV(pl, dict(clf__C=Cs), cv=12)
    # estimator.fit(X, y)
    #
    # print('Best Parameter:', estimator.best_params_)
    # print('Best Score:', estimator.best_score_)
    #
    # # Grid search cross validation for solver in LogisticRegression
    # solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    #
    # estimator = GridSearchCV(pl, dict(clf__solver=solvers), cv=12)
    # estimator.fit(X, y)
    #
    # print('Best Parameter:', estimator.best_params_)
    # print('Best Score:', estimator.best_score_)

    # Fit the pipeline on train data
    pl.fit(X_train, y_train)

    # Score the test data
    print('Mean Accuracy:', pl.score(X_test, y_test), '\n')

    # Print the classification report of test data
    y_pred = pl.predict(X_test)
    target_names = ['DJIA Decreased (0)', 'DJIA Increased (1)']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Print the cross tabulation table of test data
    pred = pl.predict(X_test)
    print(pd.crosstab(y_test, pred, rownames=["Actual"], colnames=["Predicted"]), '\n')

    # Print coefficient-word table
    vct = pl.get_params()['vct']
    clf = pl.get_params()['clf']
    words = vct.get_feature_names()
    coeffs = clf.coef_.tolist()[0]
    coeff_df = pd.DataFrame({'Word': words,
                             'Coefficient': coeffs})
    coeff_df = coeff_df.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
    print(coeff_df.head(15), '\n')
    print(coeff_df.tail(15), '\n')
