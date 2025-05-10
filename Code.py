import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_excel("HousePricePrediction.xlsx")

#print(dataset.head(5))

shape = dataset.shape
print(shape)

obj = ( dataset.dtypes =='object')
object_cols = list(obj[obj].index)
print("catagorical variables:", len(object_cols))

int_ = (dataset.dtypes =='int')
num_cols = list(int_[int_].index)
print("integer variables:", len(num_cols))

fl= (dataset.dtypes == 'float')

fl_cols = list(fl[fl].index)
print("Float variables:", len(fl_cols))


numerical_dataset = dataset.select_dtypes(include=['number'])

plt.figure(figsize=(12,6))

sns.heatmap (numerical_dataset.corr(),
             cmap= 'BrBG',
                   fmt = '.2f',
                   linewidths = 2,
                   annot = True)

unique_values = []
for col in object_cols:
    unique_values.append(dataset[col].unique().size)
plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols, y=unique_values)

plt.figure(figsize=(18,36))
plt.title("Categorical Features: Destribution")
plt.xticks(rotation = 90)
index = 1

for col in object_cols:
    y= dataset[col].value_counts()
    plt.subplot(11,4,index)
    plt.xticks(rotation = 90)
    sns.barplot(x=list(y.index), y=y)
    index +=1


dataset.drop(['Id'],
    axis=1,
    inplace= True)

dataset['SalePrice'] = dataset['SalePrice'].fillna(
    dataset['SalePrice'].mean()
)

new_dataset= dataset.dropna()

new_dataset.isnull().sum()

from sklearn.preprocessing import OneHotEncoder

s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variavbles:")
print(object_cols)
print('No. Of. categorical features:',
      len(object_cols))

OH_encoder = OneHotEncoder(sparse_output= False, handle_unknown='ignore')
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.colums = OH_encoder.get_feature_names_out()
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis = 1)

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X = df_final.drop(['SalePrice'], axis = 1)
Y = df_final['SalePrice']
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X,Y,train_size=0.8, test_size=0.2, random_state=0
)
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error

from sklearn.ensemble import RandomForestRegressor


X_train.columns = X_train.columns.astype(str)
X_valid.columns = X_valid.columns.astype(str)
model_SVR = svm.SVR()
model_SVR.fit(X_train,Y_train)
Y_pred = model_SVR.predict(X_valid)

print(mean_absolute_percentage_error(Y_valid, Y_pred))

mode_RFR = RandomForestRegressor(n_estimators=10)
mode_RFR.fit(X_train,Y_train)
Y_pred = model_RFR.predict(X_valid)

mean_absolute_percentage_error(Y_valid, Y_pred)

from sklearn.linear_model import LinearRegression

model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)
print(mean_absolute_percentage_error(Y_valid, Y_pred))