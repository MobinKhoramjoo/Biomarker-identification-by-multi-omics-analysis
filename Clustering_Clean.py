#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd # To read data files
import numpy as np # Process matrices
from matplotlib import pyplot as plt # Plot figures
from sklearn.decomposition import PCA # Perform PCA
from sklearn.cluster import KMeans # Perform K-Means Clustering
from sklearn.preprocessing import StandardScaler # To scale the data to have 0 mean and 1 variance
# Following packages are used to build the autoencoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import models,Sequential


# In[ ]:


# Read data file
df=pd.read_csv('Raw Values_log.csv',header=0)


# In[ ]:


# Extract Long and Acute covid values 
X_acute=df.iloc[28:145,5:].values
X_long=df.iloc[145:,5:].values
X=X_long-X_acute
scaler=StandardScaler()
X=scaler.fit_transform(X) # Normalize the data
print(X.shape)


# In[ ]:


# Build AutoEncoder Architecture
model=Sequential()
model.add(Dense(100,input_shape=(782,),activation='sigmoid'))
model.add(Dense(70,activation='sigmoid'))
model.add(Dense(50,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Dense(50,activation='sigmoid'))
model.add(Dense(70,activation='sigmoid'))
model.add(Dense(100,activation='sigmoid'))
model.add(Dense(782))
model.summary()


# In[ ]:


callback=tf.keras.callbacks.EarlyStopping(monitor ='loss', min_delta = 1E-7, patience= 1000, restore_best_weights= True) #Earlystopping
model.compile(loss='mse',optimizer='adam') # Compile model with MSE loss
model.fit(X,X,epochs=1000,verbose=1,callbacks =[callback]) # Fit model for 1000 epochs


# In[ ]:


Encoder=keras.Model(model.inputs,model.layers[3].output) #Extract the bottleneck layer to give lower dimensional features
Encoder.summary() # Print summary of model structure


# In[ ]:


X_low=Encoder.predict(X) # Generate lower dimensional features


# In[ ]:


#Create label vector and color vector for plotting
label_dict_comb={'recovered':'green','mild':'blue','sever':'red'}
class_dict_comb={'recovered':0,'mild':1,'sever':2}
cvec=[label_dict_comb[label] for label in df.iloc[28:145,2].values]
org_label=[class_dict_comb[label] for label in df.iloc[28:145,2].values]


# In[ ]:


kmeans_comb=KMeans(n_clusters=3).fit(X_low) # Perform Kmeans clustering with 3 clusters
cluster_new =kmeans_comb.labels_ #Store cluster labels for each sample


# In[ ]:


keras.models.save_model(model,'AutoEnc_Allfeatures_Delta_Sigmoid.hp5',save_format='h5') # Save AutoEncoder Model


# In[ ]:


#Save Clustered Data

dict={}
for i,key in enumerate(df.keys()[1:]):
    dict[key]=X[:,i]
dict['Cluster Label']=kmeans_comb.labels_
df_new=pd.DataFrame(dict)
df_new.to_csv('Delta_Sigmoid_NoImput_Encoding_Clusters.csv')


# In[ ]:


# Plotting clustering results
s=np.ones(X.shape[0])*200
mpl.rcParams['figure.dpi']=1200
mpl.rcParams.update({'font.size':22})
colors=ListedColormap(['#69B0F8','#FEE0D2','#F2757B'])
colors1=ListedColormap(['magenta','yellow','black'])
fig,ax=plt.subplots(1,2,figsize=(20,20))
scatter=ax[0].scatter(X_low[:,0],X_low[:,1],s=s,c=org_label,cmap=colors,edgecolors='black')
ax[0].set_xlabel('Encoded Dim 1')
ax[0].set_ylabel('Encoded Dim 2')
ax[0].legend(handles=scatter.legend_elements()[0], labels=["Recovered",'Mild','Severe'],fontsize=35,markerscale=5)
scatter1=ax[1].scatter(X_low[:,0],X_low[:,1],s=s,c=cluster_new,cmap=colors1,edgecolors='black')
ax[1].set_xlabel('Encoded Dim 1')
ax[1].set_ylabel('Encoded Dim 2')
ax[1].legend(handles=scatter1.legend_elements()[0], labels=["Cluster A",'Cluster B', "Cluster C"],fontsize=35,markerscale=5)


# In[ ]:


# Rest of the code is for Imputation trials


# In[ ]:


#Function to impute missing values with uniform distribution
def impute_data(acute,long):
    for i in range(acute.shape[1]):
        mini_a=np.nanmin(acute.iloc[:,i].values)
        mini_l=np.nanmin(long.iloc[:,i].values)
        acute.iloc[:,i] = acute.iloc[:,i].fillna(np.random.uniform(0,mini_a))
        long.iloc[:,i] = long.iloc[:,i].fillna(np.random.uniform(0,mini_l))
        X_acute = acute.iloc[:,:].values
        X_long =long.iloc[:,:].values
        X_acute = np.log(X_acute+1E-6)
        X_long =np.log(X_long+1E-6)
        X= X_long-X_acute
    return pd.DataFrame(X,columns=acute.keys())


# In[ ]:


centroid=pd.read_csv('Sigmoid_Centroid.csv',header=0,index_col=0).values  # Read file containing centroids of original clustering


# In[ ]:


# Function to map centroids of new clusters to old clusters
def change_cluster_labels(clusters,klabels,centroid,kcentroid):
    distance=np.zeros((3,3))
    cluster_mod=np.zeros(klabels.shape)
    for i in range(3):
        for j in range(3):
            distance[i,j]=math.dist(centroid[i,:],kcentroid[j,:])       
    ids=np.argmin(distance,axis=0)
    if np.unique(ids).shape != ids.shape:
        return None
    for i,val in enumerate(klabels):
        if val==0:
            cluster_mod[i]=ids[0]
        elif val==1:
            cluster_mod[i]=ids[1]
        elif val ==2:
            cluster_mod[i]=ids[2]
        else:
            print('Error')
    return cluster_mod


# In[ ]:


# Function to find features with deviations above 65%
def find_deviations(df_clustered,cutoff,name):
    
    means =df_clustered.abs().mean(numeric_only=True)
    cluster_means =df_clustered.abs().groupby(['Cluster Labels']).mean(numeric_only=True)
    deviation=cluster_means.sub(means[:-1]).div(means[:-1])
    features=deviation.copy()
    features[features.abs()>=cutoff]=1
    features[features.abs()<cutoff]=0
    pd.concat([features,deviation.multiply(100)]).transpose().to_csv('Impute_Test_781/Features/Random_Impute_'+name+'.csv')
    
    return features


# In[ ]:


df=pd.read_csv('Cleaned_data.csv',header=0) # Read Cleaned Data
model=keras.models.load_model('AutoEnc_Allfeatures_Delta_Sigmoid.hp5') # Load trained AE model
acute = df.iloc[:117,4:].reset_index(drop=True)
long = df.iloc[117:,4:].reset_index(drop=True)
cutoff=0.65
confidence = pd.DataFrame(0, index=[0,1,2], columns = df.keys()[4:]) # Initalize new sheet to store confidence
trial=0
while trial<100: # Run 100 trials
    name ='Trial_'+str(trial+1)
    data = impute_data(acute,long) #Impute for each trial
    data.to_csv('Impute_Test/DATA/Random_Impute_'+name+'.csv')
    
    #Scale Imputed Data
    X=data.values
    scaler=StandardScaler()
    X=scaler.fit_transform(X) #Normalize data
    
    # Build Auto-encoder and obtain lower dimension data
    Encoder=keras.Model(model.inputs,model.layers[3].output)
    X_low= Encoder(X) # Get lower dimension feature for new imputed data
    
    # Run K-Means Clustering
    print('Clustering for ', name)
    kmeans_comb=KMeans(n_clusters=3).fit(X_low) # Cluster using Kmeans
    klabels=kmeans_comb.labels_
    kcentroids=kmeans_comb.cluster_centers_ #Obtain centroids
    
    # Change Labels
    
    cluster_mod = change_cluster_labels(cluster_min,klabels,centroid,kcentroids) # Map to centroids of old clustering
    if cluster_mod is None:
        print('No match found, restarting ',name)
        continue
    #Store new clustering for imputed data    
    df_clustered = pd.DataFrame(X,columns=df.keys()[4:])
    df_clustered['Cluster Labels'] =cluster_mod
    df_clustered.to_csv('Impute_Test/Clustered/Random_clustering_'+name+'.csv')
    df_labels[name]=cluster_mod
    df_labels.to_csv('Impute_Test/Clustering Labels.csv')
    
    
    #Find features in each cluster
    print('Finding features for',name)
    features = find_deviations(df_clustered,cutoff,name)
    
    confidence = confidence.add(features)
    trial=trial+1
    
    
confidence.transpose().to_csv('Impute_Test/Confidence_Feature.csv')


# In[ ]:





# In[ ]:




