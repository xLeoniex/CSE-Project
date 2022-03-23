import numpy as np
import pandas as pd

# importing module
import sys
  
# append your relative path!!
sys.path.append('./')
sys.path.append('../visual/')
#private modules
import tensornet as ttn
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split


#load data
df = pd.read_csv("cone_data_linear.csv", dtype=np.float32)


#shuffle data 
df = df.sample(frac=1, random_state=42)

#separate labels
X,y = df.iloc[:,1:], df["0"]

#get sizes
nlabels = len(np.unique(y))
nsites = 26*26

#define train size
#ndata = 13000

y=np.array(y.values, dtype=np.int32, order='C')

#split and cast to right format
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#x_train = X.iloc[:ndata]
#y_train = np.array(y.iloc[:ndata].values, dtype=np.int32, order='C')
#x_test = X.iloc[ndata:]
#y_test = np.array(y.iloc[ndata:].values, dtype=np.int32, order='C')


#be sure to have output folder
#!mkdir -p sim1
out_path = './sim2/'
in_path = './sim2/'

ndata = int(len(x_train))
#define model
model = ttn.TensorNetworkFit('C', nsites, ndata, nlabels, cutoff=10,
                             in_folder=in_path, out_folder=out_path, 
                             read_existing_tree=False,
                             sweep_number=2, f_map_dimen=4, 
                             f_map='Spin', unsupervised_on=True,
                             optimizer=ttn.ConjugateGradient(n_pass=20, l2_reg=0.0, stop_crit=10**-6),
                             verbosity=2, explain_on=False)
#put the relative path to interface.so!!
model.set_so_file("/home/ul/ul_student/ul_tfg93/frontend-v1.2/frontend/")

#data is already mapped, define array of feature dimensions
fdim = np.array([4 for i in range(nsites)], dtype=np.int32, order='C')

#TRAIN---------------------------------------------------------------------------------------------
model.train(x_train, y_train, fdim)

#PREDICT-------------------------------------------------------------------------------------------
predictions_linear = model.predict(x_test, y_test, fdim)
#predictions = model.predict(x_test, y_test, fdim)

pred = [np.argmax([p for p in pp]) for pp in predictions_linear]

#confusion matrix
conf_matrix_near = metrics.confusion_matrix(y_test,pred)
sns.heatmap(conf_matrix_near, annot = True)
plt.savefig("/pfs/data5/home/ul/ul_student/ul_tfg93/frontend-v1.2/visual/confusion_linear")
#def confusionmatrix_linear(predictions_linear, labels):
 #   label_pd = pd.Series(labels, name='actual class')
  #  predict_pd = pd.Series(predictions_linear, name='predicted class')
   # return pd.crosstab(label_pd, predict_pd)
    
#confusionmatrix_linear(pred,y.iloc[ndata:].values)
