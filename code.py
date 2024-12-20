"""

This is the python code for the study 

Improving Clinical Decisions in IR: 
Interpretable Machine Learning Models for Predicting Ascites Improvement after Transjugular Intrahepatic Portosystemic Shunt Procedures
of TIPS Ascites Prediction published at JVIR. 
doi: 10.1016/j.jvir.2024.09.022.


Thanks for reviewing the code, please do not hesitate to contact me if you have questions!

"""



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import cross_val_score, cross_validate
import warnings
warnings.filterwarnings('ignore')

np.float = float

df = pd.read_excel("XXXX.xlsx")

x = df1.iloc[:,:-1]
y = df1.iloc[:,-1]

# Train-Test Splitting

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x,y,
                                                 test_size= 0.33, random_state= 42)

# Under-Over Sampling

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
over = SMOTE(sampling_strategy= 0.7, random_state=42, n_jobs = -1)
under = RandomUnderSampler(sampling_strategy= 0.9, random_state=42)


X_sm, y_sm = over.fit_resample(X_train,y_train)
X_train, y_train = under.fit_resample(X_sm, y_sm)

# Preprocessing
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

def Standardize(X_train,X_test, discretize = True, n_bins = 10, encode = "ordinal"):
       
    disc = KBinsDiscretizer(n_bins= n_bins, encode = encode)
    sc = StandardScaler()
    temp_train = sc.fit_transform(X_train)
    temp_test = sc.transform(X_test)
    
    if discretize:
        temp_train_disc = disc.fit_transform(temp_train)
        X_train = pd.DataFrame(temp_train_disc, columns = X_train.columns)
        temp_test_disc = disc.transform(temp_test)
        X_test = pd.DataFrame(temp_test_disc, columns = X_train.columns )
        
        return X_train, X_test

    else:
        X_train = pd.DataFrame(temp_train, columns = X_train.columns)
        X_test = pd.DataFrame(temp_test, columns = X_train.columns)
        return X_train, X_test

# Feature Selection

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def feature_selector(X_train,X_test, y_train, model = 'lasso', 
                     k_features = "best", forward = True, floating = True, scoring = "roc_auc", cv = 5):
    

    if model == 'lasso':
        lasso = LogisticRegression(penalty = "l1", C = 1, solver = "liblinear",
                                   random_state = 42)

        sfs1 = SFS(lasso, k_features = k_features, forward = forward, floating = floating,
                   cv = cv, scoring = scoring, n_jobs = -1, verbose = 0)


        sfs1.fit(X_train,y_train)
        
    
    elif model == 'SVM':
        svm_sfs = SVC()

        sfs1 = SFS(svm_sfs, k_features = k_features, forward = forward, floating = floating,
                   cv = cv, scoring = scoring, n_jobs = -1, verbose = 0)


        sfs1.fit(X_train,y_train)
    
    temp = sfs1.transform(X_train)
    X_train = pd.DataFrame(temp, columns = sfs1.k_feature_names_)
    
    temp2 = sfs1.transform(X_test)
    X_test = pd.DataFrame(temp2, columns = sfs1.k_feature_names_)
    
    return X_train, X_test , sfs1

from sklearn.metrics import classification_report , confusion_matrix, roc_auc_score 
from sklearn.svm import SVC

# SVM Model Building

def SVM_model(X_train, X_test, y_train, y_test, 
              C = 1, kernel = 'rbf',
              gamma = 'auto', prob = True):
    
    svm1 = SVC(C = C, kernel = kernel, gamma = gamma,
               probability = prob, random_state = 42)


    svm1.fit(X_train,y_train)

    print("********TRAIN**********")
    print("******** SVM ********")
    print(classification_report(y_train, svm1.predict(X_train)))
    print("********** SVM Confusion Matrix ***********")
    print(confusion_matrix(y_train, svm1.predict(X_train)))
    print("********* SVM AUC Score ************")
    print(roc_auc_score(y_train,svm1.predict(X_train)))
    print("*********TEST TEST *********")

    print("******************")
    print("******** SVM ********")
    print(classification_report(y_test, svm1.predict(X_test)))
    print("********** SVM Confusion Matrix ***********")
    print(confusion_matrix(y_test, svm1.predict(X_test)))
    print("********* SVM AUC Score ************")
    auc_score = roc_auc_score(y_test,svm1.predict(X_test))
    print(f"Test Set AUC score: {auc_score}")
    
    return svm1 , auc_score

# Feature PreProcessing

X_train,X_test = Standardize(X_train, X_test, discretize = False)
X_train,X_test,sfs = feature_selector(X_train, X_test, y_train, 
                                      k_features="parsimonious", forward = False, 
                                      floating= False, scoring = "roc_auc")


# BayesianSearch

from skopt import BayesSearchCV, gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

space_svm = [Real(.5,13, name = "C"),
         Categorical(["auto","scale"], name = "gamma")]

def objective_svm(params):
    C, gamma  = params
    
    
    model = SVC(C = C, kernel = "rbf", gamma = gamma, random_state = 42)
    
    scores = cross_val_score(model, X_train,y_train, cv=5, n_jobs=-1)
    return -np.mean(scores)

n_calls = 150
random_state = 42

result_svm = gp_minimize(objective_svm, space_svm, n_calls = n_calls,
                     random_state = random_state)

# SVM Eval

C, gamma = result_svm.x

svm , auc_svm = SVM_model(X_train, X_test, y_train, y_test, C = C, 
                      kernel = "rbf", gamma = gamma)


# CatBoost Build
from catboost import CatBoostClassifier

space_cat = [Integer(3,11, name = "depth"),
         Integer(1,10, name = "l2_leaf_reg"),
         Categorical(["SymmetricTree", "Lossguide"], name = "grow_policy"),
         Real(0.2,1, name = "colsample_bylevel")]

def objective_cat(params):
    depth, l2_leaf_reg, grow_policy, colsample_bylevel  = params
    
    
    model = CatBoostClassifier(iterations = 300, depth = depth, random_seed = 42,
                         l2_leaf_reg = l2_leaf_reg , grow_policy = grow_policy,
                         learning_rate = 0.2,
                         min_data_in_leaf=1, 
                         verbose = 100,
                         colsample_bylevel = colsample_bylevel, 
                         scale_pos_weight = 1,
                         early_stopping_rounds = 250)
    
    scores = cross_val_score(model, X_train,y_train, cv=5, n_jobs=-1)
    return -np.mean(scores)

result_cat = gp_minimize(objective_cat, space_cat, n_calls = n_calls,
                     random_state = random_state)


# CatBoost Eval

best_depth, best_l2_leaf_reg, best_grow_policy, best_colsample_bylevel = result_cat.x

best_cat = CatBoostClassifier(iterations = 300, depth = best_depth, random_seed = 42,
                     l2_leaf_reg = best_l2_leaf_reg , grow_policy = best_grow_policy,
                     learning_rate = 0.2,
                     min_data_in_leaf=1, 
                     verbose = 100,
                     colsample_bylevel = best_colsample_bylevel, 
                     scale_pos_weight = 1,
                     early_stopping_rounds = 250)

best_cat.fit(X_train,y_train)

print("********TRAIN**********")
print("********CAT********")
print(classification_report(y_train, best_cat.predict(X_train)))
print(roc_auc_score(y_train,best_cat.predict(X_train)))
print("********BEST CAT TEST********")
print(classification_report(y_test, best_cat.predict(X_test)))
print("********* TEST AUC Score ************")
print(roc_auc_score(y_test,best_cat.predict(X_test)))

# Viz

#%% Viz
from sklearn.metrics import RocCurveDisplay, auc

fig, ax = plt.subplots(figsize = (10,8))

viz1 = RocCurveDisplay.from_estimator(svm, X_test, y_test,
                                      name = "SVM",
                                      alpha = .8,
                                      lw =3,
                                      color = "maroon",
                                      ax = ax)

viz2 = RocCurveDisplay.from_estimator(best_cat, X_test, y_test,
                                      name = "CatBoost",
                                      alpha = .8,
                                      lw = 3,
                                      color = "gold",
                                      ax = ax)

ax.plot([0,1],[0,1], linestyle = "--",
        lw = 2, color = "b", alpha = 0.8)

ax.set(
       xlim = [-0.05,1.05],
       ylim = [-0.05,1.05])


ax.set_xlabel("1-Specificity", fontsize = 15)
ax.set_ylabel("Sensitivity", fontsize = 15)
plt.legend(loc = "lower right")
#plt.savefig("ROC Curves.tiff", format = "tiff", dpi = 300)
plt.show()


#Interpretability Analysis with SHAP

import shap

# SVM SHAP
svm_explainer = shap.KernelExplainer(svm.predict_proba, X_train)

svm_shap_values = svm_explainer.shap_values(X_test)


#CatBoost SHAP

cat_explainer = shap.Explainer(best_cat)

cat_shap_values = cat_explainer(X_test)

# Visualization

plt.figure()

plt.title('Support Vector Machine Feature Importance\n', fontsize = 14)

shap.summary_plot(svm_shap_values[1], X_test,plot_type = 'dot',
                   color = 'green', show = False)

#plt.savefig('SVM SHAP.tiff', dpi = 600, format = 'tiff')

plt.show()


# Thanks for reviewing the code. Hope you enjoyed!


