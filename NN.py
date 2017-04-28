# ---------------------------------------------------------------------------
#   Name: Joshua Isaacson
#   Created on: April 22, 2017
#   Assignment: Q320 Final Project
#
#   Resources
#   ---------
#   * train.csv from Kaggle
#   * test.csv from Kaggle
#   * data_analysis.py
# ---------------------------------------------------------------------------


#  --- import libraries ---
import pandas
import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plot
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

#  --- set pandas output display parameters ---
pandas.set_option('display.width', 1000)


#  --- read the dataset ---
trainingUrl = 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/3136/train.csv?sv=2015-12-11&sr=b&sig=3yhBr%2FUG8%2BuS%2Bejua8KVtYsLV1WEmKP%2BG0j%2BuJDs3f0%3D&se=2017-04-30T21%3A00%3A05Z&sp=r'
train_df = pandas.read_csv(trainingUrl, dtype={"Age": numpy.float64},)

testingUrl = 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/3136/test.csv?sv=2015-12-11&sr=b&sig=coI58NKMUZZ%2FcqZahWG4t0xbBIbdcLj2NsMl710bQYY%3D&se=2017-05-01T07%3A55%3A10Z&sp=r'
test_df = pandas.read_csv(testingUrl, dtype={"Age": numpy.float64},)


# ---------------------------------------------------------------------------

#  PRE-PROCESSING

# ---------------------------------------------------------------------------


#  --- drop values that won't be useful to analysis
train = train_df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
test  = test_df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

#  --- add a variable in Sex where 0 is female, 1 is male
df_sex=pandas.get_dummies(train['Sex'],drop_first=True)
train=train.join(df_sex)

df_sex_2=pandas.get_dummies(test['Sex'],drop_first=True)
test=test.join(df_sex_2)

#  --- P class variables ---
df_pclass=pandas.get_dummies(train['Pclass'],prefix='Class').astype(int)
train=train.join(df_pclass)

df_pclass_2=pandas.get_dummies(test['Pclass'],prefix='Class').astype(int)
test=test.join(df_pclass_2)

#  --- Create random ages based on avg, std, and null freq ---
avg_age_train=train['Age'].mean()
std_age_train=train['Age'].std()
nans_age_train=train['Age'].isnull().sum()

avg_age_test=test['Age'].mean()
std_age_test=test['Age'].std()
nans_age_test=test['Age'].isnull().sum()

rand_1 = numpy.random.randint(avg_age_train-std_age_train,
                              avg_age_train+std_age_train,
                              size=nans_age_train)
rand_2 = numpy.random.randint(avg_age_test-std_age_test,
                              avg_age_test+std_age_test,
                              size=nans_age_test)

#  --- Fill null spots ---
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)

# correlation matrix measures correlation coefficient: linear dependence between two variables
corrmat=train[['Survived','Class_1','Class_2','Class_3','SibSp','Parch','Fare','male','Age']].corr()
#print(corrmat)

#  --- fill nulls with S (highest frequency) ---
train['Embarked']=train['Embarked'].fillna('S')
test['Embarked']=test['Embarked'].fillna('S')

df_em=pandas.get_dummies(train['Embarked'],prefix='Embarked')

#  --- fill null spot in Fare ---
test["Fare"].fillna(test["Fare"].median(), inplace=True)


# ---------------------------------------------------------------------------

#   CLASSIFIER IMPLEMENTATION

# ---------------------------------------------------------------------------


#  --- training and validation sets ---
# features used to train: Pclass, Fare, male or not, and Age
X_train = train[['Pclass','Fare','male','Age']]

# label used to train: Survived
Y_train = train[["Survived"]]

# features used to test: Pclass, Fare, male or not, and Age
X_test  = test[['Pclass','Fare','male','Age']]

#  --- Multi-Layer Perceptron (MLP) ---
mlp = MLPClassifier(solver='lbfgs',
                    alpha=1e-6,
                    hidden_layer_sizes=(100),
                    random_state=numpy.random.randint(0,10000),
                    learning_rate_init=0.001,
                    max_iter=10000,
                    early_stopping=False)

mlp.fit(X_train, Y_train.values.ravel())

Y_pred = mlp.predict(X_test)
score = (mlp.score(X_train, Y_train))
print("Accuracy of Multi-Layer Perceptron Predictions on the data was: {0}".format(score))


# ---------------------------------------------------------------------------

#   MODEL VALIDATION

# ---------------------------------------------------------------------------


X, y = X_train, Y_train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
mlp.fit(X_train, y_train)

fpr, tpr, _ = roc_curve(y_test, mlp.predict_proba(X_test)[:,1])

roc_auc = auc(fpr, tpr)
print('ROC AUC: %0.2f' % roc_auc)
plot.figure()
plot.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plot.plot([0, 1], [0, 1], 'k--')
plot.xlim([0.0, 1.0])
plot.ylim([0.0, 1.05])
plot.xlabel('False Positive Rate')
plot.ylabel('True Positive Rate')
plot.title('ROC Curve')
plot.legend(loc="lower right")
plot.show()


# ---------------------------------------------------------------------------

#   Kaggle Submission

# ---------------------------------------------------------------------------


submission = pandas.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": Y_pred
})
submission.to_csv('titanic.csv', index=False)