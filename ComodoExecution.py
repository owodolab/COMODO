import numpy as np
#from morphSimilarity_png import comodo
import numpy as np
import matplotlib.pyplot as plt
import math 
import pylab
import pandas as pd
import random
from scipy.stats import linregress
from sklearn.manifold import MDS
import sklearn
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 500

from cycler import cycler
COLORS =['#1b7837','#bd0026','#2c7fb8','#253494','#542788','#c51b7d']#['#ffffcc','#a1dab4','#41b6c4','#225ea8']
default_cycler = cycler(color=COLORS)
plt.rc('axes', prop_cycle=default_cycler) 
from sklearn.cluster import DBSCAN
from scipy.integrate import simps
from numpy import trapz
import time

from morphSimilarity import comodo

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))






################20 replicas###########
DistanceTFSurfacetoVolumeExpanded = np.array(comodo(r'C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Expanded Data\TFexpandeddata'))#, signature_function='shape_ratio_sig' , visualize_graphs=False))
DistanceMASSurfacetoVolumeExpanded = np.array(comodo(r"C:/Users/parth/Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Expanded Data\MASexpandeddata"))#, signature_function='shape_ratio_sig' , visualize_graphs=False))
DistanceEASSurfacetoVolumeExpanded = np.array(comodo(r"C:/Users/parth/Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Expanded Data\EASexpandeddata"))#, signature_function='shape_ratio_sig' , visualize_graphs=False))



DistanceTFAspectRatioExpanded = np.array(comodo(r'C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Expanded Data\TFexpandeddata', signature_function='shape_ratio_sig' , visualize_graphs=False))
DistanceMASAspectRatioExpanded = np.array(comodo(r"C:/Users/parth/Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\\Expanded Data\MASexpandeddata", signature_function='shape_ratio_sig' , visualize_graphs=False))
DistanceEASAspectRatioExpanded = np.array(comodo(r"C:/Users/parth/Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Expanded Data\EASexpandeddata", signature_function='shape_ratio_sig' , visualize_graphs=False))





############100replicas(400 total)######################



DistanceMASstv100replicasE = np.array(comodo(r'C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\100 replicas\MAS'))
DistanceEASstv100replicasE = np.array(comodo(r'C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\100 replicas\EAS'))


DistanceMASAR100replicasE = np.array(comodo(r'C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\100 replicas\MAS',signature_function='shape_ratio_sig'))
DistanceEASAR100replicasE = np.array(comodo(r'C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\100 replicas\EAS',signature_function='shape_ratio_sig'))


############200replicas(800 total)######################



DistanceMASAR200replicasE =np.array(comodo(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\200replicas\MAS", signature_function='shape_ratio_sig' , visualize_graphs=False))
DistanceEASAR200replicasE =np.array(comodo(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\200replicas\EAS", signature_function='shape_ratio_sig' , visualize_graphs=False))



DistanceMASstv200replicasE = np.array(comodo(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\200replicas\MAS"))
DistanceEASstv200replicasE = np.array(comodo(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\200replicas\EAS"))


##########1000Replicas
DistanceMASSV1000replicasE =np.array(comodo(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\1000replicas\MAS"))

DistanceMASAR1000replicasE =np.array(comodo(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\1000replicas\MAS",signature_function='shape_ratio_sig'))


DistanceMASAR1000replicasE =np.array(comodo(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\1000replicas\MAS",signature_function='shape_ratio_sig'))




###########################MDS######################################


from scipy.io import savemat




Distance=distmat

fig = plt.figure()
fig.patch.set_facecolor('white')
embedding = MDS(n_components=2, dissimilarity='pre_d')
X_transformed = embedding.fit_transform(Distance[:])


plt.scatter(X_transformed[0:100,0],X_transformed[0:100,1] )
plt.scatter(X_transformed[100:200,0],X_transformed[100:200,1] )
plt.scatter(X_transformed[200:300,0],X_transformed[200:300,1] )
plt.scatter(X_transformed[300:400,0],X_transformed[300:400,1] )


plt.rc('font', size=12) 
plt.legend(['GrainSize(20,40)', 'GrainSize(40,20)', 'GrainSize(40,40)','GrainSize(80,80)'])#,'Outgroup');
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')


######################NORMALIZED MATRIX######################



NorTFVSExpanded =  NormalizeData(DistanceTFSurfacetoVolumeExpanded)
NorMASVSExpanded=  NormalizeData(DistanceMASSurfacetoVolumeExpanded)
NorEASVSExpanded=  NormalizeData(DistanceEASSurfacetoVolumeExpanded)

ASNorTFExpanded=  NormalizeData(DistanceTFAspectRatioExpanded)
ASNorMASExpanded=  NormalizeData(DistanceMASAspectRatioExpanded)
ASNorEASExpanded=  NormalizeData(DistanceEASAspectRatioExpanded)




NorMASVS100replicas= NormalizeData(DistanceMASstv100replicasE)
NorEASVS100replicas=  NormalizeData(DistanceEASstv100replicasE)

ASNorMAS100replicas=  NormalizeData(DistanceMASAR100replicasE)
ASNorEAS100replicas=  NormalizeData(DistanceEASAR100replicasE)






NorMASVS200replicas=  NormalizeData(DistanceMASstv200replicasE)
NorEASVS200replicas=  NormalizeData(DistanceEASstv200replicasE)

ASNorMAS200replicas=  NormalizeData(DistanceMASAR200replicasE)
ASNorEAS200replicas=  NormalizeData(DistanceEASAR200replicasE)





SVS=NormalizeData(DistanceMASstv100replicasE)
SAR=NormalizeData(DistanceMASAR100replicasE)



#############RAND INDEX###########



tfmasexplabels=np.loadtxt('TFMASLABELSEXPANDED.txt')
easexpandedlabels=np.loadtxt('EASLABELSEXPANDED.txt')

tfmasexplabels100replicas=np.loadtxt('TFMASLABELS100replicas.txt')
clusters4100replicas=np.loadtxt('100replicas4cluster.txt')
easexpandedlabels100replicas=np.loadtxt('EASLABELS100replicas.txt')

tfmasexplabels200replicas=np.loadtxt('TFMASLABELS200replicas.txt')
tfmasexplabels200replicas4clusters=np.loadtxt('MAS200replicas4clusterslabels.txt')

easexpandedlabels200replicas=np.loadtxt('EASLABELS200replicas.txt')

tfnewdata200replicaslables=np.loadtxt('TFnewdata200replicas.txt')
SDL=np.loadtxt('SDlables.txt')
data1000cluster4=np.loadtxt('labels1000pts4cluster.txt')
data500cluster4=np.loadtxt('500replicaslabels.txt')
data700cluster4=('700replicaslabels.txt')
data400cluster4=np.loadtxt('400replicaslabels.txt')
data225cluster4=np.loadtxt('225replicaslabels.txt')
data275cluster4=np.loadtxt('275replicaslabels.txt')
MDlabels=np.loadtxt('MD500replicaslabels.txt')
MAs16replicaslabels=np.loadtxt('MASlabels16replicas.txt')

gofmatTFVS=np.array([])
gofmatTFAR=np.array([])
gofmatTFTPC=np.array([])
gofmatTFDES=np.array([])
#gof=np.zeros([100])
j=0
DistanceVS=SVS
DistanceAR=SAR
a=np.linspace(0.00000001,1,10000)
for i in range (10000):#0,10000
    dbVS=DBSCAN(eps=a[i], metric='precomputed').fit(DistanceVS)
    dbAR=DBSCAN(eps=a[i], metric='precomputed').fit(DistanceAR)
    labelsVS = dbVS.labels_
    labelsAR = dbAR.labels_
    gofVS=sklearn.metrics.adjusted_rand_score(tfmasexplabels100replicas,labelsVS)
    gofAR=sklearn.metrics.adjusted_rand_score(clusters4100replicas,labelsAR)
    
plt.plot(a,gofmatTFVS)
plt.plot(a,gofmatTFAR)
plt.xlabel('Cutoff distance')
plt.ylabel('Rand Index')
plt.ylim(0,1)

plt.rc('font', size=12) 
plt.legend(['Surface/Volume','Aspect Ratio'])


areaSV = trapz(gofmatTFVS, a)
areaAR = trapz(gofmatTFAR, a)
areaTPC = trapz(gofmatTFTPC,a)

print("Area under the curve Surface to Volume=", areaSV)
print("Area under the curve Aspect Ratio=", areaAR)





from sklearn import metrics
labels_true=clusters4100replicas
############################# Compute DBSCAN
db = DBSCAN(eps=a[514], metric='precomputed').fit(mat)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"    % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(mat, labels))

LABEL_COLOR_MAP = {-1:'#35978f',
                   0 :'#d73027',
                   1 :'#fc8d59',
                   2:'#1b7837',
                   3:'#762a83',
                   4:'#998ec3',
                   5:'#542788',
                   6:'#7fbf7b',
                   7:'#af8dc3',
                   8:'#b35806',
                   9:'#80cdc1',
                   
                   }


label_color = [LABEL_COLOR_MAP[l] for l in labels]


fig = plt.figure()
fig.patch.set_facecolor('white')
embedding = MDS(n_components=2, dissimilarity='precomputed')
X_transformed = embedding.fit_transform(Distance[:])

plt.scatter(X_transformed[:,0], X_transformed[:,1],  c=label_color)
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
cmap=('label_color')


