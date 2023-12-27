#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd # To read data files
import numpy as np  # To perform matrix operations
import math # Mathematical operations (check if nan)
import os # To access directories
from sklearn.preprocessing import StandardScaler # Normalize data
from sklearn.linear_model import LogisticRegression # Linear classifier
#Obtain metrics to test classifier
from sklearn.metrics import f1_score,confusion_matrix,roc_curve,ConfusionMatrixDisplay,auc,balanced_accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV # Split data to train-test, perform Grid search
from sklearn.feature_selection import SequentialFeatureSelector,RF # Feature elimination tools
import matplotlib.pyplot as plt # To plot data


# In[ ]:


# Open data file and split into different sets containing specific omic profiles
df1=pd.read_csv('Raw Values_log.csv',header=0)
X_acute=df1.iloc[28:145,5:].values
X_long=df1.iloc[145:,5:].values
X_acute_cyt=df1.iloc[28:145,5:52].values
X_long_cyt=df1.iloc[145:,5:52].values
X_acute_pro=df1.iloc[28:145,52:213].values
X_long_pro=df1.iloc[145:,52:213].values
X_acute_meta=df1.iloc[28:145,213:].values
X_long_meta=df1.iloc[145:,213:].values
X_acute_cp=np.hstack((X_acute_cyt,X_acute_pro))
X_long_cp=np.hstack((X_long_cyt,X_long_pro))
X_acute_cm=np.hstack((X_acute_cyt,X_acute_meta))
X_long_cm=np.hstack((X_long_cyt,X_long_meta))
X_acute_pm=np.hstack((X_acute_pro,X_acute_meta))
X_long_pm=np.hstack((X_long_pro,X_long_meta))


# In[ ]:


#Generate color vector and label vector for classifier
label_dict_comb={0:'green',1:'red'}
class_dict_comb={'Event-Free':0,'With Event':1}
cvec=[label_dict_comb[label] for label in df1.iloc[:,1].values]
org_label=df1.iloc[:,1].values


# In[ ]:


# Create Data Matrix combining all data
DATA=[]
DATA.append(X_acute)
DATA.append(X_acute_cyt)
DATA.append(X_acute_pro)
DATA.append(X_acute_meta)
DATA.append(X_acute_cp)
DATA.append(X_acute_cm)
DATA.append(X_acute_pm)
DATA.append(X_long)
DATA.append(X_long_cyt)
DATA.append(X_long_pro)
DATA.append(X_long_meta)
DATA.append(X_long_cp)
DATA.append(X_long_cm)
DATA.append(X_long_pm)
DATA.append(X_long-X_acute)
NAMES=['Acute','Acute_Cyt','Acute_Pro','Acute_Meta','Acute_CP','Acute_CM','Acute_PM','Long','Long_Cyt',
       'Long_Pro','Long_Meta','Long_CP','Long_CM','Long_PM','Delta']


# In[ ]:


#Function to perform 5 fold cross validation
def cross_validation(model, _X, _y, _cv=5):
      
      _scoring = ['balanced_accuracy', 'precision', 'recall', 'f1']
      results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
      return {"Training Accuracy scores": results['train_balanced_accuracy'],
              "Mean Training Accuracy": results['train_balanced_accuracy'].mean()*100,
              "Training Precision scores": results['train_precision'],
              "Mean Training Precision": results['train_precision'].mean(),
              "Training Recall scores": results['train_recall'],
              "Mean Training Recall": results['train_recall'].mean(),
              "Training F1 scores": results['train_f1'],
              "Mean Training F1 Score": results['train_f1'].mean(),
              "Validation Accuracy scores": results['test_balanced_accuracy'],
              "Mean Validation Accuracy": results['test_balanced_accuracy'].mean()*100,
              "Validation Precision scores": results['test_precision'],
              "Mean Validation Precision": results['test_precision'].mean(),
              "Validation Recall scores": results['test_recall'],
              "Mean Validation Recall": results['test_recall'].mean(),
              "Validation F1 scores": results['test_f1'],
              "Mean Validation F1 Score": results['test_f1'].mean()
              }


# In[ ]:


#Function to plot 5 fold cross validation results
def plot_result(x_label, y_label, plot_title, train_data, val_data,name):
        '''Function to plot a grouped bar chart showing the training and validation
          results of the ML model in each fold after applying K-fold cross-validation.
         Parameters
         ----------
         x_label: str, 
            Name of the algorithm used for training e.g 'Decision Tree'
         y_label: str, 
            Name of metric being visualized e.g 'Accuracy'
         plot_title: str, 
            This is the title of the plot e.g 'Accuracy Plot'        
         train_result: list, array
            This is the list containing either training precision, accuracy, or f1 score.       
         val_result: list, array
            This is the list containing either validation precision, accuracy, or f1 score.
         Returns
         -------
         The function returns a Grouped Barchart showing the training and validation result
         in each fold.
        '''      
        # Set size of plot
        plt.figure(figsize=(12,6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels[:len(train_data)]))
        ax = plt.gca()
        plt.ylim(0.0, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels[:len(train_data)])
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.savefig('MinimalPanel_New/RFE/FU_2023/Minimal/IMAGES/Cross_Val/'+str(name)+'_CV.png')
        plt.show()
        


# In[ ]:


# Fit classifier after performing grid search, for all molecules 
FPR=[]
TPR=[]
FPR_ALL=[]
TPR_ALL=[]
results_df=pd.DataFrame()
for i in range(len(DATA)):
    X=DATA[i]
    results_dict={}
    #GRID SEARCH
    X_train,X_test,y_train,y_test = train_test_split(X,org_label,test_size=0.1, random_state=42)
    parameters={'class_weight':[{0:1,1:1},{0:1,1:2},{0:1,1:5},{0:1,1:10},{0:2,1:1},{0:10,1:1},'balanced'],'C':[1E-5, 1E-3, 0.1, 1, 10, 100,1000]}
    _scoring = ['balanced_accuracy', 'precision', 'recall', 'f1']
    #Perform Grid search of hyperparameters
    clf = GridSearchCV(LogisticRegression(penalty='l2',max_iter=5000,), parameters,scoring=_scoring,refit='balanced_accuracy')
    clf.fit(X_train,y_train);
    results_dict = clf.best_params_
    if clf.best_params_['class_weight'] !='balanced':
        clf.best_params_['class_weight']
        dummy=str(clf.best_params_['class_weight'][0])+','+str((clf.best_params_['class_weight'][1]))
        results_dict['class_weight'] = dummy
    Clf=clf.best_estimator_
    Clf_result=cross_validation(Clf,X_train,y_train,5)
    plot_result('Linear Classifier '+ str(NAMES[i]),"Accuracy","Balanced Accuracy scores in 5 Folds",
                        Clf_result["Training Accuracy scores"],Clf_result["Validation Accuracy scores"],NAMES[i])
   
    # FIT BEST REGRESSOR FROM GRID SEARCH  
    Clf.fit(X_train,y_train)
    y_pred=Clf.predict(X_test)
    y_score=Clf.decision_function(X_test)
    y_train_pred = Clf.predict(X_train)
    results_dict['Test Accuracy'] = balanced_accuracy_score(y_test,y_pred) # Test Accuracy
    results_dict['Test F1 Score']=f1_score(y_test,y_pred) # Test F1 Score
    results_dict['Train Accuracy'] = balanced_accuracy_score(y_train,y_train_pred) # Train accuracy
    results_dict['Train F1 Score']=f1_score(y_train,y_train_pred) # Train F1 score
    results_dict['Dataset'] = NAMES[i]
    
    #PRINT CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred, labels=Clf.classes_) # Confusion matrix 
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=Clf.classes_)
    disp.plot()
    plt.savefig('MinimalPanel_New/RFE/FU_2023/IMAGES/Confusion_Matrix/'+str(NAMES[i])+'_CM.png')
    print('Saving results for ',NAMES[i])
    results_df=pd.concat([results_df,pd.DataFrame(results_dict,index=[0])],ignore_index=True)
    results_df.to_csv('MinimalPanel_New/RFE/FU_2023/Summary_new.csv')
    
    # COMPUTE ROC CURVE FOR TEST DATA
    fpr,tpr,_ =roc_curve(y_test,y_score)
    y_score=Clf.decision_function(X)
    
    # COMPUTE AND PLOT ROC CURVE FOR TEST DATA           #### Copy pasted up until hereeeeeeee #####
    fpr_all,tpr_all,_ =roc_curve(org_label,y_score)
    area=auc(fpr_all,tpr_all)s
    plt.figure()
    lw=3
    label ='Linear '+NAMES[i]+' (area= %0.2f)'
    plt.plot(fpr_all,tpr_all,color='blue',lw=lw,label=label % area,)
    plt.plot([0,1],[0,1],color='black',lw=lw,linestyle="--")
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves All Data')
    plt.legend(loc='lower right')
    plt.savefig('MinimalPanel_New/RFE/FU_2023/IMAGES/ROC/'+str(NAMES[i])+'_ROC.png')
    
    # SAVE FPR AND TRP FOR FURTHER PLOTTING
    FPR.append(fpr)
    TPR.append(tpr)
    FPR_ALL.append(fpr_all)
    TPR_ALL.append(tpr_all)
    pd.DataFrame(FPR).to_csv('MinimalPanel_New/RFE/FU_2023/FPR.csv')
    pd.DataFrame(TPR).to_csv('MinimalPanel_New/RFE/FU_2023/TPR.csv')
    pd.DataFrame(FPR_ALL).to_csv('MinimalPanel_New/RFE/FU_2023/FPR_ALL.csv')
    pd.DataFrame(TPR_ALL).to_csv('MinimalPanel_New/RFE/FU_2023/TPR_ALL.csv')
    
    # SAVE WEIGHTS AND BIAS FOR LINEAR MODEL
    Weights_dict={}
    Weights_dict['Features']=np.array(mols_list[i])
    Weights_dict['Weights']=Clf.coef_[0]
    Weights_df=pd.DataFrame(Weights_dict)
    pd.concat([Weights_df,pd.DataFrame({'Features':'Bias','Weights':Clf.intercept_[0] },index =[0])],ignore_index=True).to_csv('MinimalPanel_New/RFE/FU_2023/Weights/'+NAMES[i]+'.csv')
    


# In[ ]:


# To perform feature elimination across different sets of molecules

FPR=[]
TPR=[]
FPR_ALL=[]
TPR_ALL=[]
results_df=pd.DataFrame()
for i in range(len(DATA)):
    X=DATA[i]
    results_dict={}
    #GRID SEARCH
    parameters={'class_weight':[{0:1,1:1},{0:1,1:2},{0:1,1:5},{0:1,1:10},{0:2,1:1},{0:10,1:1},'balanced'],'C':[1E-5, 1E-3, 0.1, 1, 10, 100,1000]}
    _scoring = ['balanced_accuracy', 'precision', 'recall', 'f1']
    clf = GridSearchCV(LogisticRegression(penalty='l2',max_iter=5000,), parameters,scoring=_scoring,refit='balanced_accuracy')
    clf.fit(X_train,y_train);
    results_dict = clf.best_params_
    if clf.best_params_['class_weight'] !='balanced':
        clf.best_params_['class_weight']
        dummy=str(clf.best_params_['class_weight'][0])+','+str((clf.best_params_['class_weight'][1]))
        results_dict['class_weight'] = dummy
        
        
    #  Feature Elimination    
    Estimator=clf.best_estimator_
    selector=RFE(estimator,step=1,n_features_to_select=20)
    selector.fit(X,org_label)
    mask=selector.support_
    
    # Reduce Input Feature Space
    X_low=selector.transform(X)
    X_train,X_test,y_train,y_test,train_id,test_id = train_test_split(X_low,org_label,sample_num,test_size=0.1, random_state=42)
    Clf_result=cross_validation(Clf,X_train,y_train,5)
    plot_result('Linear Classifier '+ str(NAMES[i]),"Accuracy","Balanced Accuracy scores in 5 Folds",
                        Clf_result["Training Accuracy scores"],Clf_result["Validation Accuracy scores"],NAMES[i])
   
    # FIT BEST REGRESSOR FROM GRID SEARCH  
    Estimator.fit(X_train,y_train)
    y_pred=Estimator.predict(X_test)
    y_score=Estimator.decision_function(X_test)
    y_train_pred = Estimator.predict(X_train)
    results_dict['Test Accuracy'] = balanced_accuracy_score(y_test,y_pred) # Test accuracy
    results_dict['Test F1 Score']=f1_score(y_test,y_pred) # Test F1 score
    results_dict['Train Accuracy'] = balanced_accuracy_score(y_train,y_train_pred) # Train accuracy
    results_dict['Train F1 Score']=f1_score(y_train,y_train_pred) # Train F1 score
    results_dict['Dataset'] = NAMES[i]
    
    #PRINT CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred, labels=Estimator.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=Estimator.classes_)
    disp.plot()
    plt.savefig('MinimalPanel_New/RFE/FU_2023/Minimal/IMAGES/Confusion_Matrix/'+str(NAMES[i])+'_CM.png')
    print('Saving results for ',NAMES[i])
    results_df=pd.concat([results_df,pd.DataFrame(results_dict,index=[0])],ignore_index=True)
    results_df.to_csv('MinimalPanel_New/RFE/FU_2023/Minimal/Summary_new.csv')
    
    # COMPUTE ROC CURVE FOR TEST DATA
    fpr,tpr,_ =roc_curve(y_test,y_score)
    y_score=Estimator.decision_function(X_low)
    
    # COMPUTE AND PLOT ROC CURVE FOR TEST DATA
    fpr_all,tpr_all,_ =roc_curve(org_label,y_score)
    area=auc(fpr_all,tpr_all)
    plt.figure()
    lw=3
    label ='Linear '+NAMES[i]+' (area= %0.2f)'
    plt.plot(fpr_all,tpr_all,color='blue',lw=lw,label=label % area,)
    plt.plot([0,1],[0,1],color='black',lw=lw,linestyle="--")
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves All Data')
    plt.legend(loc='lower right')
    plt.savefig('MinimalPanel_New/RFE/FU_2023/Minimal/IMAGES/ROC/'+str(NAMES[i])+'_ROC.png')
    
    # SAVE FPR AND TRP FOR FURTHER PLOTTING
    FPR.append(fpr)
    TPR.append(tpr)
    FPR_ALL.append(fpr_all)
    TPR_ALL.append(tpr_all)
    pd.DataFrame(FPR).to_csv('MinimalPanel_New/RFE/FU_2023/Minimal/FPR.csv')
    pd.DataFrame(TPR).to_csv('MinimalPanel_New/RFE/FU_2023/Minimal/TPR.csv')
    pd.DataFrame(FPR_ALL).to_csv('MinimalPanel_New/RFE/FU_2023/Minimal/FPR_ALL.csv')
    pd.DataFrame(TPR_ALL).to_csv('MinimalPanel_New/RFE/FU_2023/Minimal/TPR_ALL.csv')
    
    # SAVE WEIGHTS AND BIAS FOR LINEAR MODEL
    Weights_dict={}
    Weights_dict['Features']=np.array(mols_list[i])[mask]
    Weights_dict['Weights']=Estimator.coef_[0]
    Weights_df=pd.DataFrame(Weights_dict)
    pd.concat([Weights_df,pd.DataFrame({'Features':'Bias','Weights':Estimator.intercept_[0] },index =[0])],ignore_index=True).to_csv('MinimalPanel_New/RFE/FU_2023/Minimal/Weights/'+NAMES[i]+'.csv')
    

    
    
    


# In[ ]:


# The code below is for plotting ROC Curves

AREA_ALL=[]
for i in range(len(FPR_ALL)):
    AREA_ALL.append(auc(FPR_ALL[i],TPR_ALL[i]))  #Find area under ROC Curve


# In[ ]:


# Plot ROC curves for all omicsm Cytokine, metabolites, Proteins and minimal panel
plt.figure(figsize=(12,12))
lw=3
plt.plot(FPR_ALL[7],TPR_ALL[7],color='blue',lw=lw,label='All-Omics (AUC= %0.2f)' % AREA_ALL[7],)
plt.plot(FPR_ALL[8],TPR_ALL[8],color='#FF8000',lw=lw,label=' Cyto (AUC= %0.2f)' % AREA_ALL[8],)
plt.plot(FPR_ALL[9],TPR_ALL[9],color='#008000',lw=lw,label=' Pro (AUC= %0.2f)' % AREA_ALL[9],)
plt.plot(FPR_ALL[10],TPR_ALL[10],color='#8000FF',lw=lw,label=' Meta (AUC= %0.2f)' % AREA_ALL[10],)
plt.plot(fpr_all,tpr_all,color='red',lw=lw,label='Minimal  (AUC= %0.2f)' % area,)
plt.plot([0,1],[0,1],color='black',lw=lw,linestyle="--")
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves All Data')
plt.legend(loc='lower right')
plt.show()


# In[ ]:


# To plot prediction score for a given estimator

y_test_score=estimator.decision_function(X_test)
y_hat_test=1/(1 + np.exp(-y_test_score)) # Obtain sigmoid of prediction value of the classifier for the data
sample_test=np.array(range(X_test.shape[0]))+1 # Get number of sample for plotting
colors=ListedColormap(['green','red']) # Color map for differetn classes
s_test=np.ones(X_test.shape[0])*200 # Marker size for each sample
plt.figure(figsize=(15,15)) 
scatter=plt.scatter(y_hat_test,sample_test,s=s_test,c=y_test,cmap=colors,edgecolors='black') #Scatter plot of prediciton scores
plt.axvline(x=0.5,color='black',lw=3,linestyle="--") # Classification threshold
plt.xlabel('Prediction Score',fontsize=30) 
plt.ylabel('Sample number',fontsize=30)
plt.xlim([-0.1,1.1])
plt.legend(handles=scatter.legend_elements()[0], labels=["Event Free",'With Event'],fontsize=32,loc='lower left',markerscale=5)

