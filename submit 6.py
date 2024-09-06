import numpy as np
import sklearn
from scipy.linalg import khatri_rao

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# X_train has 32 columns containing the challeenge bits
	# y_train contains the responses
	
	# THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
	# If you do not wish to use a bias term, set it to 0
     from sklearn.linear_model import LogisticRegression #Importing the linear model from sklearn library, in this case: Logistic Regression
     transformed_x_train = my_map(X_train) #Converting the input into a form compatble with a linaer model using the my_map function 
     best_model = LogisticRegression(C=1000, tol=0.001, penalty='l2') #Defining the parameters of the model (the parameters were decided upon, keeping in my mind the higest accuracy on provided test set, along with the time take to train)
     best_model.fit( transformed_x_train,y_train ) #Training the model on the transfored train data set 
     w = best_model.coef_[0] if best_model.coef_ is not None else 0 #Extracting the feature matrix of the trained model
     b = best_model.intercept_[0] if best_model.intercept_ is not None else 0  #Extracting the intercept of the trained model
     return w, b


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
    from scipy.linalg import khatri_rao
    x = X.copy()
    d = 1-np.dot(2, x) # Updating the value of set bit, so that, now every ith set bei is either +1 or -1
    x = np.transpose(x)
    d = np.transpose(d)
    for i in range(32): # Calculating xi=di*d(i+1).....d(n)
         if i==0:
              x[31-i]= d[31-i]
         else:
              x[31-i]= d[31-i]*x[32-i]
    x = np.transpose(x)
    d = np.transpose(d)
    r = khatri_rao(np.transpose(x),np.transpose(x)) #using khatri rao in a row wise manner
    r = np.transpose(r)
    indices_to_remove = [] #Removing the repeating indices 
    for i in range(32):
          for j in range(0, i+1):
                indices_to_remove.append(32*i+j)
    mask = np.logical_not(np.isin(np.arange(1024), indices_to_remove)) #extracting the needed expression of the input 
    l = r[:, mask]
    feat = np.concatenate((l, x), axis=1) 
    return feat #returning the created mask
