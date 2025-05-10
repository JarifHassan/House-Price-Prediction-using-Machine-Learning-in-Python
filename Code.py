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

