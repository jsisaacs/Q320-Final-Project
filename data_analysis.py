# ---------------------------------------------------------------------------
#   Name: Joshua Isaacson
#   Created on: April 22, 2017
#   Assignment: Q320 Final Project
#
#   Resources
#   ---------
#   * train.csv from Kaggle
#   * test.csv from Kaggle
#   * NN.py
# ---------------------------------------------------------------------------


#  --- import libraries ---
from pandas import Series, DataFrame
import pandas
import matplotlib.pyplot as plot
import numpy
from sklearn import datasets, preprocessing, metrics, model_selection
from sklearn.neural_network import MLPClassifier


#  --- set pandas output display parameters ---
pandas.set_option('display.width', 1000)


#  --- read the dataset ---
trainingUrl = 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/3136/train.csv?sv=2015-12-11&sr=b&sig=3yhBr%2FUG8%2BuS%2Bejua8KVtYsLV1WEmKP%2BG0j%2BuJDs3f0%3D&se=2017-04-30T21%3A00%3A05Z&sp=r'
df_train = pandas.read_csv(trainingUrl, dtype={"Age": numpy.float64},)


# ---------------------------------------------------------------------------

#   DATA EXPLORATION

# ---------------------------------------------------------------------------


#  --- shape ---
# returns 892 instances and 12 instances
print("Data Shape")
print(df_train.shape)
print("")

#  --- head ---
print("Raw Data (First 20 Lines)")
print(df_train.head(20))
print("")

#  --- descriptions ---
print("Data Description")
print(df_train.describe())
print("")

#  --- mean survival rate ---
print("Mean Survival Rate")
print(df_train['Survived'].mean())
print("")

print("Data Count")
print(df_train.count())
print("")

# Distribution of Survival
plot.figure(figsize = (6,4))
fig, axis = plot.subplots()
df_train.Survived.value_counts().plot(kind = 'barh', color = 'blue', alpha = 0.65)
axis.set_ylim(-1, len(df_train.Survived.value_counts()))
plot.title("Distribution of Survival")
plot.show()

# Distribution of Age Inside Class
df_train.Age[df_train.Pclass == 1].plot(kind = 'kde')
df_train.Age[df_train.Pclass == 2].plot(kind = 'kde')
df_train.Age[df_train.Pclass == 3].plot(kind = 'kde')
plot.title("Distribution of Age Inside Class")
plot.xlabel("Age")
plot.legend(('Class 1', 'Class 2', 'Class 3'), loc = 'best')
plot.show()
