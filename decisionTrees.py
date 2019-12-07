"""
Authors:
Date:
Description:
"""

import numpy as np
from util import *
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt

import graphviz

def main() :
    #load data
    train_data, test_data = load_data('spambase/spambase.data')

    print('Using Decision Trees ...')

    print('\n')
    print('Params: entropy criterion - best splitter')
    clf = DecisionTreeClassifier(criterion = "entropy", splitter = "best")
    clf = clf.fit(train_data.X, train_data.y)
    print('Training Accuracy: ', clf.score(train_data.X,train_data.y))
    print('Testing Accuracy: ', clf.score(test_data.X,test_data.y))

    print('\n')
    print('Params: entropy criterion - random splitter')
    clf = DecisionTreeClassifier(criterion = "entropy", splitter = "random")
    clf = clf.fit(train_data.X, train_data.y)
    print('Training Accuracy: ', clf.score(train_data.X,train_data.y))
    print('Testing Accuracy: ', clf.score(test_data.X,test_data.y))

    print('\n')
    print('Params: gini criterion - best splitter')
    clf = DecisionTreeClassifier(criterion = "gini", splitter = "best")
    clf = clf.fit(train_data.X, train_data.y)
    print('Training Accuracy: ', clf.score(train_data.X,train_data.y))
    print('Testing Accuracy: ', clf.score(test_data.X,test_data.y))

    print('\n')
    print('Params: gini criterion - random splitter')
    clf = DecisionTreeClassifier(criterion = "gini", splitter = "random")
    clf = clf.fit(train_data.X, train_data.y)
    print('Training Accuracy: ', clf.score(train_data.X,train_data.y))
    print('Testing Accuracy: ', clf.score(test_data.X,test_data.y))

    print('\n')
    print('Params: entropy criterion - best splitter - max depth')
    training_accuracies = []
    testing_accuracies = []
    max_depth =[]
    for i in range(1,50):
        max_depth.append(i)
        clf = DecisionTreeClassifier(criterion = "entropy", splitter = "best", max_depth = i)
        clf = clf.fit(train_data.X, train_data.y)
        training_accuracy = clf.score(train_data.X,train_data.y)
        testing_accuracy = clf.score(test_data.X,test_data.y)
        training_accuracies.append(training_accuracy)
        testing_accuracies.append(testing_accuracy)
        print('Max Depth: ', i)
        print('Training Accuracy: ',training_accuracy)
        print('Testing Accuracy: ',testing_accuracy)

    """plt.plot(max_depth,training_accuracies,'bo-',label ='training accuracy')
    plt.plot(max_depth,testing_accuracies,'ro-',label ='testing accuracy')
    plt.title("Accuracy of Decision Trees vs Maximum Depth of the Tree")
    plt.legend()
    plt.xlabel("Maximum Depth")
    plt.ylabel("Accuracy")
    plt.savefig('DecisionTreesAccuracy')
    plt.show()"""

    #confusion matrix
    clf = DecisionTreeClassifier(criterion = "entropy", splitter = "best")
    clf = clf.fit(train_data.X, train_data.y)
    prediction = clf.predict(test_data.X)

    confusion_matrix = np.zeros((2,2))
    accuracy =0
    for i in range(len(prediction)):
        if  prediction[i] == 0 and test_data.y[i] == 0:
            confusion_matrix[0][0] +=1
            accuracy +=1
        elif prediction[i] == 1 and test_data.y[i] == 1:
            confusion_matrix[1][1] +=1
            accuracy +=1
        elif prediction[i] == 0 and test_data.y[i]  == 1:
            confusion_matrix[1][0] +=1

        elif prediction[i] == 1 and test_data.y[i]  == 0:
            confusion_matrix[0][1] +=1

    # outputting confusion matrix
    print('\n')
    print('Confusion Matrix')
    print(' prediction')
    print('   -1  1')
    print('   -----')
    print('-1| '+ str(int(confusion_matrix[0][0])) + '  ' + str(int(confusion_matrix[0][1])))
    print(' 1| '+ str(int(confusion_matrix[1][0])) + '  ' + str(int(confusion_matrix[1][1])))
    print('\n')









if __name__ == "__main__" :
    main()
