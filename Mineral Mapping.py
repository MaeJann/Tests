#%% Load Modules

import cv2
import numpy as np
import pandas as pd 

#%% Load Data


# --- Load coordinates and labels 
data = pd.DataFrame(pd.read_csv('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Extracted/Observations_Composite.csv', header=None, sep = ';' ))  
data = data.as_matrix()
x,y = data[:, 1], data[:, 2]
labels = data[:,3]

# --- define quadratic slide around coordinates 
extend = 20
x_min = x - extend 
x_max = x + extend
y_min = y - extend
y_max = y + extend

# --- Load maximum intensity images:
max_ppl = cv2.imread('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Images/plane/Naxos_Boudin_8_ppl_0.jpg')
max_xpl = cv2.imread('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Images/processed/MaxInt.jpg')


#%% Functions for feature extraction

# Extract average color value for each channel around the coordinates:
def extract_col_val(image, x_min,x_max, y_min, y_max):
    col_vals = np.zeros((len(x_min), 3), dtype="uint8")
    for i in range(len(x_min)):
        col_vals[i, 0:3] = np.array([np.mean(image[y_min[i]:y_max[i], x_min[i]:x_max[i],0]), np.mean(image[y_min[i]:y_max[i], x_min[i]:x_max[i], 1]), np.mean(image[y_min[i]:y_max[i], x_min[i]:x_max[i], 2])])   
    return col_vals

def comp_std(image, x_min,x_max, y_min, y_max):
    std_vals = np.zeros((len(x_min), 3), dtype="uint8")
    for i in range(len(x_min)):
        std_vals[i, 0:3] = np.array([np.std(image[y_min[i]:y_max[i], x_min[i]:x_max[i],0]), np.std(image[y_min[i]:y_max[i], x_min[i]:x_max[i], 1]), np.std(image[y_min[i]:y_max[i], x_min[i]:x_max[i], 2])])
    return std_vals



#%% extract individual features

PPL_Color = extract_col_val(max_ppl, x_min,x_max, y_min, y_max)
XPL_Color = extract_col_val(max_xpl, x_min,x_max, y_min, y_max)

PPL_std = comp_std(max_ppl, x_min,x_max, y_min, y_max)
XPL_std = comp_std(max_xpl, x_min,x_max, y_min, y_max)

#%% Combine extracted features to feature matrix

X = np.hstack((PPL_Color, XPL_Color, PPL_std, XPL_std))


#%% Quick Feature Test

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X ,labels, test_size=0.4, random_state = 1) 

knn3 = KNeighborsClassifier(n_neighbors = 3)
knn3.fit(X_train, y_train)
y_pred = knn3.predict(X_test)
print()
print()
print("Classification Accuracy KNN: (K = 3):", int(metrics.accuracy_score(y_test, y_pred)*100), "%")

logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print("Classification Accuracy Logistic Regression:", int(metrics.accuracy_score(y_test, y_pred)*100), "%")


supvecm= svm.SVC(kernel='poly', degree=3)
supvecm.fit(X_train,y_train)
y_pred = supvecm.predict(X_test)
print("Classification Supported Vector Machines:", int(metrics.accuracy_score(y_test, y_pred)*100), "%")

#%% 


import matplotlib.pyplot as plt


im_coord= pd.DataFrame(pd.read_csv('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Mineral Mapping/Testarea 2/testarea_coord2.csv', header=None, sep = ';' ))  
im_coord  = im_coord.as_matrix()


x_range= np.arange(np.min(im_coord[:, 1]), np.max(im_coord[:, 1]),1)
y_range = np.arange(np.min(im_coord[:, 2]), np.max(im_coord[:, 2]),1)

testarea = max_ppl[y_range[0]:y_range[-1], x_range[0]:x_range[-1], :]

plt.imshow(testarea)

x_coord= np.zeros((len(x_range)*len(y_range)))
y_coord= np.zeros((len(x_range)*len(y_range)))

k = 0
for j in range(len(y_range)):
    for i in range(len(x_range)):
        x_coord[k] = x_range[i]
        y_coord[k] = y_range[j]
        k+= 1

extend = 10
x_min = x_coord - extend 
x_max = x_coord + extend
y_min = y_coord - extend
y_max = y_coord + extend

PPL_Color = extract_col_val(max_ppl, x_min,x_max, y_min, y_max)
XPL_Color = extract_col_val(max_xpl, x_min,x_max, y_min, y_max)

PPL_std = comp_std(max_ppl, x_min,x_max, y_min, y_max)
XPL_std = comp_std(max_xpl, x_min,x_max, y_min, y_max)

X_new = np.hstack((PPL_Color, XPL_Color, PPL_std, XPL_std))






#%%
y_new = supvecm.predict(X_new)

#%%
Map = y_new.reshape((len(y_range),len(x_range)))
#%%


Map[Map == 1] = 0
Map[Map  == 2] = 70
Map[Map  == 3] = 150
Map[Map  == 4] = 200
Map[Map  == 5] = 250

plt.imshow(Map, cmap = "gray")
plt.savefig("Map")
#%%
)
