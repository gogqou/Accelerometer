'''
Created on Nov 11, 2015

@author: gogqou
'''


import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split,StratifiedKFold
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import re
import os
import scipy.signal as signal
import random
from scipy.stats import skew,kurtosis,mode, pearsonr
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import scipy.cluster.hierarchy as sch
from matplotlib.patches import Rectangle
import os.path as path
sns.set()

fs = 32.0
def load_data(base_path,filter=[],downsample=1):
    fs = 32.0/downsample
    activities = [item for item in os.listdir(base_path) if os.path.isdir(base_path+item) and ('MODEL' not in item)]
    raw = []
    for activity in activities:
        if activity not in filter:
            for item in os.listdir(base_path+activity):
                data = raw_to_mps(np.loadtxt(base_path+activity+'/'+item))
                data = data[::downsample,:]
                sex, iden = re.findall('-([mf])([0-9]+)\.txt',item)[0]
                raw += [{'data':data,'sex':sex,'iden':sex+iden,'activity':activity}]
    random.shuffle(raw)
    return raw

def raw_to_mps(data):
    g = 9.81
    return -1.5*g+data*3*g/63

def myfft(signl,fs=fs):
    w = np.fft.rfftfreq(signl.shape[0],1./fs)
    signal_fft = np.fft.rfft(signl,axis=0)
    return w,np.abs(signal_fft)

def med1d(data,kernel_sz=3):
    return np.apply_along_axis(signal.medfilt,0,data,kernel_sz)

def conv1d(data,kernel_sz=11,sig=3):
    filt = signal.gaussian(kernel_sz,sig)
    return np.apply_along_axis(np.convolve,0,data,filt/np.sum(filt),mode='same')

def cc(signal1,signal2,k=0):
    signal1 = signal1[:len(signal1)-k]
    signal2 = signal2[k:]
    signal1 = signal1 - np.mean(signal1)
    signal2 = signal2 - np.mean(signal2)
    return np.mean(signal1*signal2)/(np.std(signal1)*np.std(signal2))

def ac(signal1,k=1):
    return cc(signal1,signal1,k)

def sigw(w,fft):
    return np.sum(w*fft)/np.sum(fft)
        
def log_loss(y_true,y_pred):
    eps = 1e-15
    y_pred = y_pred/y_pred.sum(axis=1)[:,np.newaxis]
    y_pred = np.maximum(eps,y_pred)
    y_pred = np.minimum(1-eps,y_pred)
    return -np.sum(np.log(y_pred[range(len(y_true)),y_true.astype(int)]))/len(y_true)

def get_features(trial, downsample=1):

    '''
    features = ('target','meanx','meany','meanz','mean',
                'varx','vary','varz','var',
                'skewx','skewy','skewz','skew',
                'kurtx','kurty','kurtz','kurt',
                'corrxy','corrxz','corryz')
    '''
    tmp = {}
    data = trial['data']
    data1d = np.sqrt((data**2).sum(axis=1)) #take magnitude of acc.
    fig=plt.figure()
    fig.add_subplot(311)
    plt.plot(data[:,0])
    fig.add_subplot(312)
    plt.plot(data[:,1])
    fig.add_subplot(313)
    plt.plot(data[:,2])
    if path.isfile('Graphs/'+trial['activity']+trial['iden']+'.png')is False:
        plt.savefig('Graphs/'+ trial['activity']+'_'+trial['iden']+'.png', format='png')
        print 'saved '+trial['activity']+'_'+trial['iden']+'.png'

    plt.close()
    tmp['target'] = trial['activity']
    data = trial['data'][::downsample,:]
    data1d = np.sqrt((data**2).sum(axis=1))
    tmp['meanx'],tmp['meany'],tmp['meanz'] = data.mean(axis=0)
    tmp['mean'] = data1d.mean()
    tmp['varx'],tmp['vary'],tmp['varz'] = data.var(axis=0)
    tmp['var'] = data1d.var()
    tmp['skewx'],tmp['skewy'],tmp['skewz'] = skew(data,axis=0)
    tmp['skew'] = skew(data1d)
    tmp['kurtx'],tmp['kurty'],tmp['kurtz'] = kurtosis(data,axis=0)
    tmp['kurt'] = kurtosis(data1d)
    tmp['corrxy'] = pearsonr(data[:,0],data[:,1])[0]
    tmp['corrxz'] = pearsonr(data[:,0],data[:,2])[0]
    tmp['corryz'] = pearsonr(data[:,1],data[:,2])[0]
    return tmp

def mycv(clf,df,cv=None,nfolds=5,w=None,N=20):
    target = df['target']
    le = LabelEncoder().fit(target)
    target = le.transform(target)
    data = np.array(df.drop('target',axis=1))
    features = np.array(df.drop('target',axis=1).columns)
    ll = []
    acc = []
    fi = []
    cm = []
    if cv is None:
        cv = StratifiedKFold(target,n_folds=nfolds)
    for train_idx, valid_idx in cv:
        Xtrain, ytrain = data[train_idx,:], target[train_idx]
        Xtest, ytest = data[valid_idx,:], target[valid_idx]
        #print Xtest
        #clf.fit(Xtrain,ytrain,sample_weight=w)
        clf.fit(Xtrain,ytrain)
        ypred = clf.predict(Xtest)
        ll += [log_loss(ytest,clf.predict_proba(Xtest))]
        acc += [accuracy_score(ytest,ypred)]
        fi += [clf.feature_importances_]
        cm += [confusion_matrix(ytest,ypred)]
    ll = np.mean(ll)
    acc,accstd = np.mean(acc),np.std(acc)
    fi = np.array(fi)
    cm = np.array(cm)
    cm,cmstd = cm.sum(axis=0),cm.std(axis=0)
    cmn = cm.astype(float)/cm.sum(axis=0)
    
    return {'ll':ll,'acc':acc,'accstd':accstd,'fi':fi,'cm':cm,'cmstd':cmstd,'cmn':cmn,'features':features,'classes':le.classes_}


def show_conf_mat(cvstats, datasetname):
    
    cmn = cvstats['cmn']
    labels = cvstats['classes']
    labels = np.array([x.replace('_',' ') for x in labels])

    cluster_threshold = .7

    ax2 = plt.gcf().add_axes([0.3,0.8,0.6,0.05])
    
    
    Z= sch.linkage(cmn,method='ward')
    Z2 = sch.dendrogram(Z,color_threshold=np.max(Z[:,2])*cluster_threshold,color_list=['black','blue','green','red'])#cluster_threshold)
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.axis('off')


    sC = np.copy(cmn)
    sC = sC[:,Z2['leaves']]
    sC = sC[Z2['leaves'],:]
    slabels = labels[Z2['leaves']]

    axmatrix = plt.gcf().add_axes([0.3,0.2,0.6,0.6])
    im = axmatrix.matshow(sC, aspect='auto', origin='lower',cmap=plt.cm.Blues)
    
    for i in range(sC.shape[0]):
            for j in range(sC.shape[1]):
                if sC[i,j] > 0.5:
                    plt.gca().annotate(str(sC[i,j])[0:4],xy=(j,i),xytext=(j-0.25,i+0.1),color='w')
                elif sC[i,j] > 0.05:
                    plt.gca().annotate(str(sC[i,j])[0:4],xy=(j,i),xytext=(j-0.25,i+0.1),color='k')
    
    axmatrix.set_yticks(np.arange(cmn.shape[0]))
    axmatrix.set_yticklabels(slabels)
    axmatrix.set_xticks(np.arange(cmn.shape[0]))
    axmatrix.xaxis.set_ticks_position('bottom')
    axmatrix.set_xticklabels(slabels,rotation=45,ha='right')
    axmatrix.set_xlabel('true')
    axmatrix.set_ylabel('predicted')
    
    plt.grid(False)


    # draw rectangles
    clusters = sch.fcluster(Z,cluster_threshold)[Z2['leaves']]
    for cluster in set(clusters):
        inds = np.where(clusters == cluster)
        xmin = np.min(inds) - 0.5
        xmax = np.max(inds) + 0.5
        axmatrix.add_patch(Rectangle(
            (xmin,xmin), # (x,y)
            xmax-xmin,   # width
            xmax-xmin,   # height
            fill=False   # remove background

        ))
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axmatrix)
    divider2 = make_axes_locatable(ax2)

    #draw colorbar
    cax = divider.append_axes("right", size="5%", pad=0.05)
    _ = divider2.append_axes("right", size="5%", pad=0.05)
    plt.axis('off')

    plt.gcf().colorbar(im,cax=cax)
    plt.savefig('Confusion Matrix'+datasetname+'.png', format='png')
def show_feat_importance(cvstats,N=20):
    #fig=plt.figure()
    fi,features = cvstats['fi'],cvstats['features']
    fi,fistd = fi.mean(axis=0),fi.std(axis=0)
    sc = max(fi)
    fi /= sc
    fistd /= sc
    srtd_idx = np.argsort(fi)[::-1]
    N = min(N,len(fi))
    pos = (np.arange(N) + 0.5)[::-1]
    _=plt.barh(pos,fi[srtd_idx[:N]],align='center')
    #plt.barh(pos,fi[srtd_idx[:N]],align='center')
    plt.yticks(pos,np.array(features)[srtd_idx[:N]])
    plt.title('Feature Importance')
    plt.savefig('Feature Importance'+'.png', format='png')
    

def main():
    base_path = '/home/gogqou/git/CDIPS-Project/HMP_Dataset/'

    group_dict = {'Brush_teeth':'Brush teeth',
              'Comb_hair':'Comb hair',
              'Use_telephone':'Use telephone',
              'Sitdown_chair':'Posture change',
              'Standup_chair':'Posture change',
              'Liedown_bed': 'Posture change',
              'Getup_bed':'Posture change',
              'Drink_glass':'Drink glass',
              'Pour_water':'Pour water',
              'Descend_stairs':'Locomotion',
              'Climb_stairs':'Locomotion',
              'Walk':'Locomotion'}

    params = {'subsample': 1.0, 'max_depth': 3, 'learning_rate': 0.1, 'loss': 'deviance'}

    
    clf = GradientBoostingClassifier(**params)
   
    trials = load_data(base_path,filter=['Eat_soup','Eat_meat'],downsample=1)
    print 'loaded trials'
    df = pd.DataFrame(get_features(trial) for trial in trials)
    print 'constructed dataframe'
    
    cvstats = mycv(clf,df,nfolds=5)
    print 'done cross validation'
    print 'Confusion Matrix:'
    print cvstats['cm']
    print 'Classes: '
    print cvstats['classes']
    
    plt.figure()
    show_conf_mat(cvstats, '_Raw')
    print 'raw confusion matrix'
    df['target'] = df['target'].apply(lambda x: group_dict[x])
    print 'grouped activities' 
    cvstats = mycv(clf,df,nfolds=5)
     
    plt.figure()
    show_conf_mat(cvstats, '_Grouped')
    print 'done cross validation'
    print 'Confusion Matrix, Grouped :'
    print cvstats['cm']
    print 'Classes: '
    print cvstats['classes']
    print 'done confusion matrix'
    plt.figure()
    show_feat_importance(cvstats)
    
    return 1
if __name__ == '__main__':
    main()