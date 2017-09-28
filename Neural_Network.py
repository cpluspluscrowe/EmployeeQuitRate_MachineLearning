from pandas import read_csv
from pandas import Categorical
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

def min_max_scale(df_column):
    new_df_column = (df_column-df_column.min())/(df_column.max()-df_column.min())
    return new_df_column

def scale_column(column_name,dataframe):
    column = df[column_name]
    new_column = min_max_scale(column)
    df[column_name] = new_column
    return df

def categorize(column_name,dataframe):
    new_column = Categorical(df[column_name]).codes
    df[column_name] = new_column
    return df

df = read_csv(r"HR_comma_sep.csv")
df = scale_column("number_project",df)
df = scale_column("average_monthly_hours",df)
df = scale_column("time_spend_company",df)
df = categorize("sales",df)
df = categorize("salary",df)

length = len(df)

data = df.loc[:,df.columns != 'left']
solution = df.loc[:,df.columns == 'left']

train_data = data[:int(length*.8)].values
train_solution = solution[:int(length*.8)].values

test_data = data[int(length*.8):].values
test_solution = solution[int(length*.8):].values

results = []

class Result():
    def __init__(self,number_of_features,number_of_layers,score,learning_rate = -1):
        self.feature_count = number_of_features
        self.layer_count = number_of_layers
        self.score = score
        self.learning_rate = learning_rate

def TestAccuracyOfModel(number_of_features,number_of_layers,learning_rate,results):
    reg = MLPClassifier(solver='lbfgs', alpha=learning_rate,hidden_layer_sizes=(number_of_features, number_of_layers), random_state=1)
    reg.fit (train_data, train_solution)

    test_prediction = reg.predict(test_data)

    score = reg.score(test_data,test_solution)
    result = Result(number_of_features,number_of_layers,score,learning_rate)
    results.append(result)
    print("Number of features: {0}; Number of Layers: {1}; Score: {2}; Learning Rate: {3}".format(number_of_features,number_of_layers,score,learning_rate))

for number_of_features in range(1,10):
    for number_of_layers in range(1,10):
        TestAccuracyOfModel(number_of_features,number_of_layers,1e-06,results) #store results in the list results

results.sort(key=lambda x: x.score, reverse=True)

#best_combos = results[:20] #first 20 best scored
#learning_rates = [1e-04,1e-05,1e-06]
#for combo in best_combos:
#    for rate in learning_rates:
#        TestAccuracyOfModel(number_of_features,number_of_layers,rate,best_combos)

#best_combos.sort(key=lambda x: x.score, reverse=True)

with open("best_scores.txt","w") as f:
    for x in results[:10]:
        f.write("Score: {0}; Number of features: {1}; Number of Layers: {2}; Learning Rate: {3}".format(x.score,x.feature_count,x.layer_count,x.learning_rate))


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.array([x.feature_count for x in results])
Y = np.array([x.layer_count for x in results])
#X, Y = np.meshgrid(X, Y)
Z = np.array([x.score for x in results])

# Plot the surface.
surf = ax.scatter(X, Y, Z,c='r')

# Customize the z axis.
ax.set_zlim(0, 1)

plt.xlabel('Feature Count')
plt.ylabel('Hidden Layer Count')
plt.title('Feature Count vs Layer Count for Model Accuracy')
plt.savefig('Neural_Network_Scores.pdf')

