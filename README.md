# Is It A Spam ?

This project aims to classify emails as spam or not spam. We use the following machine learning algorithms: Decision Trees, Naive Bayes and Neural Networks. 

# Data Pre-processing
The data has .... (state how many features and what the feature names- is data balaanced ...)
We split the data into 75% training data and 25% testing data. We then separate the features from the labels and make sure we do not look at the testing data labels.

# Decision Trees
Using the DecisionTreeClassifier in the sklearn python package, we fit the training data ans evaluate the performace of the decision tree model.
    
    clf = DecisionTreeClassifier(criterion = "entropy", splitter = "best")
    clf = clf.fit(train_data.X, train_data.y)
    score = clf.score(test_data.X,test_data.y)
    

