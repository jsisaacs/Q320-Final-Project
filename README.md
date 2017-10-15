# Titanic Analysis

In this project, I analyzed the [Titanic dataset](https://www.kaggle.com/c/titanic) and prediced the surival rates of the passengers on-board.  

### Exploratory Data Analysis
Here is a general breakdown:
![EDA](https://i.imgur.com/mTsX1T9.png)
![Distribution of Survival](https://i.imgur.com/zxRY6Be.png)
![Distribution of Age Inside Class](https://i.imgur.com/5gsJcoP.png)

### Pre-Processing
- drop values that won't be useful to analysis
- add a variable in Sex where 0 is female, 1 is male
- P class variables
- Create random ages based on avg, std, and null freq
- Fill null spots in Age
- fill null values in Embarked
- fill null spot in Fare 

### Classifier Implementation
- features used to train: Pclass, Fare, male or not, and Age
- label used to train: Survived
- Used [MLPClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

![MLP](https://i.imgur.com/g44xzkC.png)

### Model Validation

![ROC](https://i.imgur.com/0gYjhqQ.png)

1.00 is the best possible ROC AUC, higher is better.
