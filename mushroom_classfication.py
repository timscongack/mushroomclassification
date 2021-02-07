import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)



X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]
feature_names = list(mush_df[:2])
# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2


def answer_five():
    from sklearn.tree import DecisionTreeClassifier
    from adspy_shared_utilities import plot_decision_tree
    from sklearn.model_selection import train_test_split
    #Load in data and train the decision tree classifier
    #Set max tree depth to truncate tree
    clf = DecisionTreeClassifier(random_state=0).fit(X_train2, y_train2)
    #Print Accuracy of Training and Test
    #print('Accuracy of Decision Tree classifier on training set: {:.2f}'
    # .format(clf.score(X_train2, y_train2)))
    #print('Accuracy of Decision Tree classifier on test set: {:.2f}'
    # .format(clf.score(X_test2, y_test2)))

    #Combine the Feature importance with the feature names
    df = pd.merge((pd.Series(mush_df2.columns.values)[2:].reset_index()), 
                    pd.DataFrame(clf.feature_importances_), 
                    left_index=True, right_index=True).set_index('0_x').drop(['index'], axis=1).nlargest(5, "0_y").index.values.tolist()
    return df

answer_five()

def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    param_range = np.logspace(-4,1,6)
    #training the model with c=1 to allow for simpler decision fuction 
    clf = SVC(C=1).fit(X_train2, y_train2)
    #create the validation curve, removed cross-validation and using default
    train_scores, test_scores = validation_curve(SVC(), X_subset, y_subset,
                                            param_name='gamma',
                                            param_range=param_range)
    return (np.mean(train_scores,axis=1),np.mean(test_scores,axis=1))

answer_six()

