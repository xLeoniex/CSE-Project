import numpy as np
import pandas as pd

#importing module
import sys

import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
  
# append your relative path!!
sys.path.append('./')
sys.path.append('./visual/')
#private modules
import tensornet as ttn

#load data
df = pd.read_csv("cone_data_nearest.csv", dtype=np.float32)


#shuffle data 
df = df.sample(frac=1, random_state=42)

#separate labels
X,y = df.iloc[:,1:], df["0"]

#get sizes
nlabels = len(np.unique(y))
nsites = 26*26

y=np.array(y.values, dtype=np.int32, order='C')
#define train size
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#split and cast to right format
#x_train = X.iloc[:ndata]
#y_train = np.array(y.iloc[:ndata].values, dtype=np.int32, order='C')
#x_test = X.iloc[ndata:]
#y_test = np.array(y.iloc[ndata:].values, dtype=np.int32, order='C')


#be sure to have output folder
#!mkdir -p sim1
out_path = './sim1/'
in_path = './sim1/'

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
model.set_so_file("./")

#data is already mapped, define array of feature dimensions
fdim = np.array([4 for i in range(nsites)], dtype=np.int32, order='C')

#TRAIN---------------------------------------------------------------------------------------------
model.train(x_train, y_train, fdim)

#PREDICT-------------------------------------------------------------------------------------------
predictions_near = model.predict(x_test, y_test, fdim)
#predictions = model.predict(x_test, y_test, fdim)

pred = [np.argmax([p for p in pp]) for pp in predictions_near]

#predictions_near =np.array(prediction_near.values, dtype=np.int32, order='C')
#confusion matrix
conf_matrix_near = metrics.confusion_matrix(y_test,pred)
sns.heatmap(conf_matrix_near, annot = True)
plt.savefig("/pfs/data5/home/ul/ul_student/ul_tfg93/frontend-v1.2/visual/confusion_near")
#def confusionmatrix_near(predictions_near, labels):
    #label_pd = pd.Series(labels, name='actual class')
    #predict_pd = pd.Series(predictions_near, name='predicted class')
    #return pd.crosstab(label_pd, predict_pd)
    
#confusionmatrix_near(pred,y.iloc[ndata:].values)

