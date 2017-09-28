from pandas import read_csv
from pandas import Categorical
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import cross_val_score

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

reg = tree.DecisionTreeClassifier(max_depth=20)
reg.fit (train_data, train_solution)

test_prediction = reg.predict(test_data)

score = reg.score(test_data,test_solution)
print(score)

dot_data = tree.export_graphviz(reg,out_file='tree.dot')
with open("tree.dot","r") as f:
    content = f.read()

content = content.replace("X[0]","satisfaction_level")
content = content.replace("X[1]","last_evaluation")
content = content.replace("X[2]","number_project")
content = content.replace("X[3]","average_monthly_hours")
content = content.replace("X[4]","time_spend_company")
content = content.replace("X[5]","Work_accident")
content = content.replace("X[6]","promotion_last_5years")
content = content.replace("X[7]","sales")
content = content.replace("X[8]","salary")

with open("tree_revised.dot","w") as f:
    f.write(content)
