
# coding: utf-8

## Git

# In[80]:

# get_ipython().run_cell_magic(u'bash', u'', u'\n#git config --global user.email "LN"\n#git config --global user.name  "LN"\n\ngit add .ipynb_checkpoints/MIL-checkpoint.ipynb\ngit add MIL.ipynb\n\ngit commit -m \'MIL: backup\'\ngit push origin master')


# In[ ]:

# from IPython.display import display, clear_output
# clear_output()
# clear()

# In[ ]:




## Setup

# In[2]:

#import math

import collections
from collections import Counter
from sets import Set
  

import datetime
import time


import os
import subprocess


import json
import csv


from multiprocessing import Pool
import itertools
from operator import itemgetter 

from random import randint


import pylab
import matplotlib.pyplot as plt

import math 

# import mlpy

import numpy as np
import numpy.linalg as la
import pandas as pd

from numpy.lib import stride_tricks

import scipy
from scipy.signal import *
from scipy.stats import mode

from sklearn import *
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import *

# In[3]:

# sys.path.append("../../misvm/")
import misvm

reload(misvm)


# In[4]:

from IPython.display import HTML
from base64 import b64encode

# path_to_audio = "sound/done.wav"
# audio_type = "wav"

# sound = open(path_to_audio, "rb").read()
# sound_encoded = b64encode(sound)
# sound_tag = """
#     <audio id="beep" controls src="data:audio/{1};base64,{0}">
#     </audio>""".format(sound_encoded, audio_type)

# play_beep = """
# <script type="text/javascript">
#     var audio = document.getElementById("beep");
#     audio.play();
#     alert("Done")
# </script>
# """


# HTML(sound_tag) 

#HTML(play_beep) #notify when the script is done running
#HTML(alert)


alert_when_done = """
<script type="text/javascript">
    alert("Done")
</script>
"""

HTML(alert_when_done)


## GENERAL ALGORITHMS

### DF Manipulation

# In[5]:

def filter_df(df, key_value_pairs, operation='equal', return_complement=False):
    for key in key_value_pairs:
        if operation=='equal':
            selection = df[key] == key_value_pairs[key]
            
        elif operation=='not_equal':
            selection = df[key] != key_value_pairs[key]
            
        elif operation=='in':
            selection = df[key].isin(key_value_pairs[key])
            
        elif operation=='not_in':
            selection = df[key].notin(key_value_pairs[key])
    
    selected = df[selection]
    if not return_complement:
        return selected

    else:
        complement = df[selection==False]
        return selected, complement

    
#select items by their indexes
def get_items_from_list(l, indexes):
    items = itemgetter(*indexes)(l)
    
    if type(items) == tuple:
        items = list(items)
    else:
        items = [items]
    
    return items

def get_columns(df, except_columns):
    return [col for col in df.columns if col not in except_columns]


def get_features_and_labels_for_scikitlearn(df):
    df = df.dropna()
    
    feature_columns = get_columns(df, ['user','activity'])
    features = df.ix[:,feature_columns] #make sure to select columns
        
    labels = df['activity']

    return features, labels


### Evaluation

# In[6]:

def leave_one_person_out_cv(classifiers, features_and_labels):
    results = []
    
    user_ids = features_and_labels['user'].value_counts().index.values
    
    for i in range(len(user_ids)):

        
        testing_user = user_ids[i] 
        print "testing_user: ", testing_user
        
        testing_features_and_labels, training_features_and_labels = filter_df(features_and_labels, {'user': testing_user}, return_complement=True)
        
        training_features, training_labels = get_features_and_labels_for_scikitlearn(training_features_and_labels)
        testing_features, testing_labels = get_features_and_labels_for_scikitlearn(testing_features_and_labels)
        
        
        if training_features.shape[1] == 1: #handle MIL bags instead of traditional features
            training_features = [x[0] for x in training_features.values]
            testing_features = [x[0] for x in testing_features.values]

        
        for algorithm, classifier in classifiers.items(): 
            
            print algorithm
            
            classifier.fit(training_features, training_labels)
            predictions = classifier.predict(testing_features)

            results.append(pd.DataFrame({
                'prediction_score': predictions,
                'prediction': np.sign(predictions),
                'reference': testing_labels,
                'user': testing_user,
                'algorithm': algorithm,
            }))

    return pd.concat(results)

# In[7]:

def compute_true_positive(df, label, reference_column = 'reference', prediction_column = 'prediction'):
    temp = df[df[reference_column] == df[prediction_column]]
    temp = temp[temp[prediction_column] == label]
    return float(len(temp))

def compute_false_positive(df, label, reference_column = 'reference', prediction_column = 'prediction'):
    temp = df[df[reference_column] != df[prediction_column]]
    temp = temp[temp[prediction_column] == label]
    return float(len(temp))

def compute_false_negative(df, label, reference_column = 'reference', prediction_column = 'prediction'):
    temp = df[df[reference_column] != df[prediction_column]]
    temp = temp[temp[reference_column] == label]
    return float(len(temp))


def compute_precision_for_label(df, label, reference_column = 'reference', prediction_column = 'prediction'):
    tp = compute_true_positive(df, label, reference_column, prediction_column)
    fp = compute_false_positive(df, label, reference_column, prediction_column)
    
    if tp + fp == 0:
        return 0
    
    return tp / (tp + fp)

def compute_recall_for_label(df, label, reference_column = 'reference', prediction_column = 'prediction'):
    tp = compute_true_positive(df, label, reference_column, prediction_column)
    fn = compute_false_negative(df, label, reference_column, prediction_column)
    
    if tp + fn == 0:
        return 0
    
    return tp / (tp + fn)

def compute_accuracy(df, reference_column = 'reference', prediction_column = 'prediction'):
    correct_prediction = df[df[reference_column] == df[prediction_column]]    
    accuracy = len(correct_prediction) / float(len(df))
    return accuracy


def get_confusion_matrix(df, reference_column = 'reference', prediction_column = 'prediction'):

    all_labels = Set(df[reference_column].values.tolist()) | Set(df[prediction_column].values.tolist()) 
    name_to_id_mapping = list(all_labels)
    
    prediction_ids = df[prediction_column].map(lambda x: name_to_id_mapping.index(x))
    reference_ids = df[reference_column].map(lambda x: name_to_id_mapping.index(x))
    
    cm = confusion_matrix(reference_ids, prediction_ids)
    
    confusion_matrix_df = pd.DataFrame(cm, columns = name_to_id_mapping, index = ["reference_" + str(x) for x in name_to_id_mapping])
    return confusion_matrix_df

def computing_result_metrics(df):
    precision = compute_precision_for_label(df, 1)
    recall = compute_recall_for_label(df, 1)

    f1 = 2 * precision * recall / (precision + recall)

    accuracy = compute_accuracy(df)

    return pd.Series({
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            })

t = time.clock()

def time_start():
    global t
    t = time.clock()
    
def time_stop():
    print 'time duration: ', (time.clock() - t), 's'
    
# In[9]:

def plot_labels_over_time(series, figsize=(15, 5)):
    
    figure(figsize=figsize)
    fig, ax = plt.subplots()
    

    #normalize the y tick labels (make it start from 0)
    id_to_label_mapping = sorted(series.value_counts().index.tolist())
    series = series.map(lambda x: id_to_label_mapping.index(x))

    #plot the occurances
    plt.plot(series.index, series.values, 'o')
    
    
    
    #add y labels
    plt.yticks(np.arange(min(series.values)-1, max(series.values)+2, 1.0))
    
    labels = []
    for i in range(0, len(id_to_label_mapping)):
        labels.append(label_legend_df['Label name'].ix[id_to_label_mapping[i]])
    
    labels.insert(0, '') 
    ax.set_yticklabels(labels)
    


## Low-level Activities

# In[11]:

data_src = '../../_data/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'


### Data analysis

# In[12]:

#read in data
data_df = pd.read_csv(data_src, error_bad_lines=False, names=['user', 'activity', 'timestamp', 'x', 'y', 'z'])
data_df['z'] = data_df['z'].astype('string').map(lambda x: x.replace(';', '')).astype('float')

#split them into segments - people, time, activity


data_df


# In[13]:

#filter out people with missing data
users_with_data_from_all_activities = data_df.groupby('user')['activity'].value_counts().unstack().dropna().index
data_df = data_df[data_df['user'].isin(users_with_data_from_all_activities)]


# In[14]:

#initialize variables
user_ids = data_df['user'].value_counts().index.values.astype('int').tolist()
activities = data_df['activity'].value_counts().index.tolist()

print user_ids
print activities


### Baseline

#### Feature extraction

# In[15]:

#test it for one user
# one_user_data_df = filter_df(data_df, {'user': user_ids[1]})
user_data_df = data_df


# In[16]:

def compute_power_spectrum_backup(series, fs = 100.0):
    n = len(series) # length of the signal
    k = arange(n)
    T = n/fs
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = scipy.fft(series.values)/n # fft computing and normalization
    Y = Y[range(n/2)]
    
    #     Y[:2] = 0
    Y = abs(Y)
    
    
    return Y


# In[17]:

def compute_power_spectrum(series, frequency = 20):


    ps = np.abs(np.fft.fft(series))**2
    freqs = np.fft.fftfreq(series.size, 1/frequency)

    idx = np.argsort(freqs)
    idx = idx[len(idx)/2:]


#     plt.plot(freqs[idx], ps[idx])

    return ps[len(ps)/2:]


# In[18]:

sampling_frequency = 20

window_size = 5 * sampling_frequency  #number of seconds
window_step = int(window_size/2)

frequency_bins = 5 #bin in frequency bands
frequency_value_bins = 10 #for value discretization


def extract_statistical_features(series, prefix, feature_row):
    feature_row[prefix + '_mean'] = series.mean()
    feature_row[prefix + '_std'] = series.std()
    feature_row[prefix + '_var'] = series.var()
    feature_row[prefix + '_min'] = series.min()
    feature_row[prefix + '_max'] = series.max()
    feature_row[prefix + '_energy'] = np.mean(series**2)
    feature_row[prefix + '_entropy'] =  scipy.stats.entropy(np.histogram(series, bins=frequency_value_bins, density=True)[0])
    
def extract_features_of_one_column(df, column, feature_row):
    
    
    
    series = df[column]
    
    #time domain features
    extract_statistical_features(series, '1_' + column, feature_row)
    
    
    #frequency domain features
#     power_spectrum = compute_power_spectrum(series, frequency = sampling_frequency)
    
#     frequency_band_size = len(power_spectrum) / frequency_bins
#     for band in range(frequency_bins):
#         extract_statistical_features(power_spectrum[band*frequency_band_size : (band+1)*frequency_band_size], '1_' + column + '_freq' + str(band), feature_row)
    

    
def extract_features_of_two_columns(df, columns, feature_row):
    feature_row['2_' + columns[0] + columns[1] + '_correlation'] = df[columns[0]].corr(df[columns[1]])



def extract_features_in_window(df):
    
    #mean, variance
    feature_row = {}

    df['m'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    
    
    extract_features_of_one_column(df, 'x', feature_row)
    extract_features_of_one_column(df, 'y', feature_row)
    extract_features_of_one_column(df, 'z', feature_row)
    extract_features_of_one_column(df, 'm', feature_row)
    
    
    extract_features_of_two_columns(df, ['x', 'y'], feature_row)
    extract_features_of_two_columns(df, ['x', 'z'], feature_row)
    extract_features_of_two_columns(df, ['y', 'z'], feature_row)
        
    extract_features_of_two_columns(df, ['x', 'm'], feature_row)
    extract_features_of_two_columns(df, ['y', 'm'], feature_row)
    extract_features_of_two_columns(df, ['z', 'm'], feature_row)
    
    
    
    
    feature_row['user'] = df.iloc[0]['user']
    feature_row['activity'] = df.iloc[0]['activity']

    return feature_row
    
    
    
def extract_features(df):
    feature_rows = []
    for i in range(0, len(df)-window_size, window_step):
        window = df.iloc[i:i+window_size]
        
        feature_row = extract_features_in_window(window)
        feature_rows.append(feature_row)
    
    
    return pd.DataFrame(feature_rows)

#generate feature vectors
features_of_all_activities = pd.DataFrame()

data_df = data_df.dropna()

for user_id in user_ids:

    one_user_data_df = filter_df(data_df, {'user': user_id})
    
    for activity in activities:

        one_user_one_activity_data_df = filter_df(one_user_data_df, {'activity': activity})

        features_of_one_activity_df = extract_features(one_user_one_activity_data_df)
        features_of_all_activities = pd.concat([features_of_all_activities, features_of_one_activity_df])
    
feature_columns = get_columns(features_of_all_activities, ['user','activity'])
feature_columns


#### Classification

# In[19]:

classifiers = {}


# classifiers['GaussianNB'] = GaussianNB()

# classifiers['DecisionTreeClassifier'] = tree.DecisionTreeClassifier()

classifiers['RandomForestClassifier'] = RandomForestClassifier(n_estimators=5)
# classifiers['ExtraTreesClassifier'] = ExtraTreesClassifier(n_estimators=5)


# classifiers['NearestCentroid'] = NearestCentroid()
# classifiers['KNeighborsClassifier'] = KNeighborsClassifier(n_neighbors=5)

# classifiers['LogisticRegression'] = LogisticRegression()
# classifiers['LinearSVC'] = svm.LinearSVC()
# classifiers['SVC'] = svm.SVC(kernel='poly', max_iter=20000)

classifiers


#### Cross-validation

# In[20]:

#perform activity recognition - feature extraction, svm

features_and_labels = filter_df(features_of_all_activities, {'user': user_ids[0]})

features, labels = get_features_and_labels_for_scikitlearn(features_and_labels)

#very good results
for algorithm, classifier in classifiers.items():
    classification_results = cross_validation.cross_val_score(classifier, features, labels, cv=10)
    print classifier.__class__.__name__, '\t & \t', classification_results.mean().round(2), '\t & \t', classification_results.std().round(2), ' \\\\'



#### Leave-one-person-out-validation

# In[ ]:

def swap_axes(df, from_axis, to_axis):
    
    from_axis_column = '1_' + from_axis + '_'
    to_axis_column = '1_' + to_axis + '_'
    
    df = df.rename(columns=lambda c: c.replace(from_axis_column, from_axis_column + '_temp_'))
    df = df.rename(columns=lambda c: c.replace(to_axis_column, from_axis_column))
    df = df.rename(columns=lambda c: c.replace(from_axis_column + '_temp_', to_axis_column))

    return df


# In[ ]:

accuracies = {}


for algorithm, classifier in classifiers.items():

    accuracy_per_classifier = []
    #cross-validation
    for i in range(len(user_ids)):

        testing_user = user_ids[i] 

        testing_features_and_labels, training_features_and_labels = filter_df(features_of_all_activities, {'user': testing_user}, return_complement=True)

        
        #switch 2 axes
#         testing_features = swap_axes(testing_features, 'x', 'y')
#         testing_features = swap_axes(testing_features, 'z', 'y')
#         testing_features = swap_axes(testing_features, 'x', 'y')
        
        training_features, training_labels = get_features_and_labels_for_scikitlearn(training_features_and_labels)
        testing_features, testing_labels = get_features_and_labels_for_scikitlearn(testing_features_and_labels)
        
        
        
        classifier.fit(training_features, training_labels)
        predictions = classifier.predict(testing_features)
        
        accuracy = np.mean(testing_labels == predictions)
        accuracy_per_classifier.append(accuracy)
        
        print i, accuracy
#         if accuracy == nan:
#             print predictions
        
        
    accuracies[algorithm] = accuracy_per_classifier

    
    
    
for algorithm, accuracy in accuracies.items():
    print '\n%s Accuracy: %.2f%% (%.2f) (%s) ' % (algorithm, np.average(accuracy), np.std(accuracy), accuracy)


# In[ ]:

swap_axes(testing_features, 'x', 'y')[:1]


### Normalization

# In[ ]:

test_df = filter_df(data_df, {'user': user_ids[15], 'activity': activities[0]})


#### Low pass filter

# In[ ]:

window = 200 
# low_pass_filtered = test_df[['x', 'y', 'z']]
low_pass_filtered = pd.rolling_mean(test_df[['x', 'y', 'z']], window)

low_pass_filtered.plot()


# In[ ]:

magnitude = np.sqrt(np.power(low_pass_filtered['x'],2) + np.power(low_pass_filtered['y'],2) + np.power(low_pass_filtered['z'],2))

magnitude.plot()


# In[ ]:

#TODO: converting gravity into rotation matrix 

- need to just calculate the pitch, roll (no yaw - it does not matter which direction we perform the activity)
get_ipython().set_next_input(u'- so if the yaw is not normalized - what does it mean');get_ipython().magic(u'pinfo mean')


# In[ ]:

#TODO: rotate acceleration into absolute coordinate 


### MIL Prediction

#### MIL ALGORITHMS

# In[21]:

#generate randomly negative instances 
def get_random_instances(df, number_of_instances, label, label_mode = 'not_equal'):
    
    random_instances = []
    while (len(random_instances)) < number_of_instances:
        
        random_id = randint(1, len(df)) - 1
        random_instance = df.ix[random_id]
        
        if label_mode == 'not_equal' and random_instance['activity'] != label:
            random_instances.append(random_instance)
        
        elif label_mode == 'equal' and random_instance['activity'] == label:
            random_instances.append(random_instance)

            
    return pd.DataFrame(random_instances)        



#prepare MIL input: get only the features from the bags
def get_bags_data_from_dfs(bags_dfs):
    columns = get_columns(bags_dfs[0], ['index', 'activity', 'user'])
    
    bags_data_for_MIL = []
    for bag in bags_dfs:
        bags_data_for_MIL.append(bag[columns].values)
        
    return bags_data_for_MIL


#prepare MIL input: get features and also generate labels
def get_bags_and_labels_for_MIL(bags_dfs, label):
    labels = [label] * len(bags_dfs)

    return get_bags_data_from_dfs(bags_dfs), labels


#prepare MIL input: get instances for bags
def get_instances_with_labels(bags_dfs, positive_label):
    columns = get_columns(bags_dfs[0], ['index', 'activity', 'user'])
    
    all_bags = pd.concat(bags_dfs)
    
    all_bags_data = all_bags[columns].values
    instances_data = [x for x in all_bags_data]
    
    all_bags_labels = (all_bags['activity'] == positive_label).astype('int').replace(0, -1)
    instances_labels = [x for x in all_bags_labels]
    
    
    return instances_data, instances_labels



# In[22]:

def get_bags_for_user(features_of_all_activities, activity_of_interest, users, number_of_instances_per_bag, noise_level_pos, noise_level_neg = 0):

    
    #compute params
    features = filter_df(features_of_all_activities, {'user': users}, operation='in')

    all_activities_features_df = features.reset_index() 
    one_activity_features_df = filter_df(all_activities_features_df, {'activity': activity_of_interest})


    number_of_bags = int(len(one_activity_features_df) /  number_of_instances_per_bag)
    
    number_of_positive_instances_in_positive_bag = int(number_of_instances_per_bag * (1-noise_level_pos))
    number_of_negative_instances_in_positive_bag = number_of_instances_per_bag - number_of_positive_instances_in_positive_bag


    number_of_negative_instances_in_negative_bag = int(number_of_instances_per_bag * (1-noise_level_neg))
    number_of_positive_instances_in_negative_bag = number_of_instances_per_bag - number_of_negative_instances_in_negative_bag
    
    
    
    #generating noisy training data ======
    positive_bags = []
    for bag_id in range(number_of_bags):

        bag_start = bag_id * number_of_instances_per_bag
        bag_end = bag_start + number_of_positive_instances_in_positive_bag
    #     print bag_start, bag_end

        bag = one_activity_features_df[bag_start:bag_end]

        #get random number of negative instances
        negative_instances = get_random_instances(all_activities_features_df, number_of_negative_instances_in_positive_bag, activity_of_interest)
        bag = pd.concat([bag, negative_instances])

        positive_bags.append(bag)


    #make the number of negative bags the same as positive bags
    negative_bags = []
    for i in range(len(positive_bags)):
        negative_instances = get_random_instances(all_activities_features_df, number_of_negative_instances_in_negative_bag, activity_of_interest)
        positive_instances = get_random_instances(all_activities_features_df, number_of_positive_instances_in_negative_bag, activity_of_interest, 'equal')
        
        negative_bags.append(pd.concat([negative_instances, positive_instances]))
  
        
        
    print '#bags:', len(positive_bags), len(negative_bags)
    if len(positive_bags) > 0 and len(negative_bags) > 0:
        print '#instance/bag:', len(positive_bags[0]), len(negative_bags[0])

    
    return positive_bags, negative_bags


# In[32]:

#MIL
 
#   SVM     : a standard supervised SVM
#   SIL     : trains a standard SVM classifier after applying bag labels to each
#             instance
#   MISVM   : the MI-SVM algorithm of Andrews, Tsochantaridis, & Hofmann (2002)
#   miSVM   : the mi-SVM algorithm of Andrews, Tsochantaridis, & Hofmann (2002)

#   NSK     : the normalized set kernel of Gaertner, et al. (2002)
#   STK     : the statistics kernel of Gaertner, et al. (2002)

#   MissSVM : the semi-supervised learning approach of Zhou & Xu (2007)
#   MICA    : the MI classification algorithm of Mangasarian & Wild (2008)

#   sMIL    : sparse MIL (Bunescu & Mooney, 2007)
#   stMIL   : sparse, transductive  MIL (Bunescu & Mooney, 2007)
#   sbMIL   : sparse, balanced MIL (Bunescu & Mooney, 2007)


# Construct classifiers
verbose = False
kernel = 'linear'
C = 1.0

mil_classifiers = {}
    
mil_classifiers['SIL'] = misvm.SIL(kernel=kernel, C=C, verbose=verbose)

mil_classifiers['miSVM'] = misvm.miSVM(kernel=kernel, C=C, verbose=verbose)
mil_classifiers['MISVM'] = misvm.MISVM(kernel=kernel, C=C, verbose=verbose)

# mil_classifiers['MICA'] = misvm.MICA(kernel=kernel, C=C, verbose=verbose)
# mil_classifiers['MissSVM'] = misvm.MissSVM(kernel=kernel, C=C, verbose=verbose)

# mil_classifiers['sMIL'] = misvm.sMIL(kernel=kernel, C=C, verbose=verbose)
# mil_classifiers['stMIL'] = misvm.stMIL(kernel=kernel, C=C, verbose=verbose)
# mil_classifiers['sbMIL'] = misvm.sbMIL(kernel=kernel, C=C, verbose=verbose)

# mil_classifiers['NSK'] = misvm.NSK(kernel=kernel, C=C, verbose=verbose)
# mil_classifiers['STK'] = misvm.STK(kernel=kernel, C=C, verbose=verbose)



#### Simple split

# In[24]:

activity_of_interest = activities[0] 
user_of_interest = user_ids[0]

number_of_instances_per_bag = 12 * 1 
noise_level = 0.0 #0 => supervised learning

      
positive_bags, negative_bags = get_bags_for_user(features_of_all_activities, activity_of_interest, [user_of_interest], number_of_instances_per_bag, noise_level)


# In[25]:

negative_bags[0]


# In[26]:

training_rate = 0.8



positive_bags_for_MIL, positive_labels_for_MIL = get_bags_and_labels_for_MIL(positive_bags, 1)
negative_bags_for_MIL, negative_labels_for_MIL = get_bags_and_labels_for_MIL(negative_bags, -1)
print len(positive_bags_for_MIL), len(positive_labels_for_MIL)
print len(negative_bags_for_MIL), len(negative_labels_for_MIL)


training_item_count = int(len(positive_bags_for_MIL) * training_rate)



train_bags = positive_bags_for_MIL[:training_item_count] + negative_bags_for_MIL[:training_item_count]
train_labels = positive_labels_for_MIL[:training_item_count] + negative_labels_for_MIL[:training_item_count]

test_bags = positive_bags_for_MIL[training_item_count:] + negative_bags_for_MIL[training_item_count:]
test_labels = positive_labels_for_MIL[training_item_count:] + negative_labels_for_MIL[training_item_count:]

print len(train_bags), len(test_bags)
print len(train_labels), len(test_labels) 

# Train/Evaluate classifiers
accuracies = {}
for algorithm, classifier in mil_classifiers.items():
    print algorithm
    classifier.fit(train_bags, train_labels)
    predictions = classifier.predict(test_bags)
    accuracies[algorithm] = (test_labels == np.sign(predictions))
    
#     print algorithm, test_labels, predictions 

for algorithm, accuracy in accuracies.items():
    print '\n%s Accuracy: %.1f%% (%s) ' % (algorithm, 100 * np.average(accuracy), accuracy)


#### MIL Cross validation

##### Generating data for cross validation

##### MIL N-Fold Cross validation

# In[28]:

def learn_and_predict_MIL(classifier, bags_positive_train, bags_negative_train, bags_positive_test, bags_negative_test):
    
    #TRAINING - shared among both bag and instance accuracy
    bags_data_positive_train = get_bags_data_from_dfs(bags_positive_train)
    bags_data_negative_train = get_bags_data_from_dfs(bags_negative_train)

    bags_data_train = bags_data_positive_train + bags_data_negative_train
    bags_labels_train = [1] * len(bags_data_positive_train) + [-1] * len(bags_data_negative_train)


    classifier.fit(bags_data_train, bags_labels_train)


    
    
    #TESTING

    #bag accuracy ===================        
    bags_data_positive_test = get_bags_data_from_dfs(bags_positive_test)
    bags_data_negative_test = get_bags_data_from_dfs(bags_negative_test)

    bags_data_test = bags_data_positive_test + bags_data_negative_test
    bags_labels_test = [1] * len(bags_data_positive_test) + [-1] * len(bags_data_negative_test)


    prediction_scores = classifier.predict(bags_data_test)
    
    bag_prediction_df = pd.DataFrame({'prediction_score': pd.Series(prediction_scores), 'reference': pd.Series(bags_labels_test)})
    bag_prediction_df['prediction'] = bag_prediction_df['prediction_score'].map(lambda x: np.sign(x))
    


    #instance accuracy ===================
    instance_data_positive_test, instance_labels_positive_test = get_instances_with_labels(bags_positive_train, activity_of_interest)
    instance_data_negative_test, instance_labels_negative_test = get_instances_with_labels(bags_negative_train, activity_of_interest)

    instance_data_test = instance_data_positive_test + instance_data_negative_test
    instance_labels_test = instance_labels_positive_test + instance_labels_negative_test


    prediction_scores = classifier.predict(instance_data_test)
        
    instance_prediction_df = pd.DataFrame({'prediction_score': pd.Series(prediction_scores), 'reference': pd.Series(instance_labels_test)})
    instance_prediction_df['prediction'] = instance_prediction_df['prediction_score'].map(lambda x: np.sign(x))
    

    return bag_prediction_df, instance_prediction_df

    
def cross_val_score_MIL(classifier, positive_bags, negative_bags, cv_n_folds=4, shuffle=False):
    
    #output    
    bags_prediction_df = pd.DataFrame()
    instances_prediction_df = pd.DataFrame()
    
    kf = KFold(len(positive_bags), n_folds=cv_n_folds, shuffle=shuffle)
    
    for train, test in kf:
        print train, test
        

        bags_positive_train = get_items_from_list(positive_bags, train)
        bags_negative_train = get_items_from_list(negative_bags, train)

        bags_positive_test = get_items_from_list(positive_bags, test)
        bags_negative_test = get_items_from_list(negative_bags, test)
   
        
        bag_prediction_df, instance_prediction_df = learn_and_predict_MIL(classifier, bags_positive_train, bags_negative_train, bags_positive_test, bags_negative_test)
        
        
        bags_prediction_df = pd.concat([bags_prediction_df, bag_prediction_df])
        instances_prediction_df = pd.concat([instances_prediction_df, instance_prediction_df])
        
        
    return bags_prediction_df, instances_prediction_df
    


    


# In[30]:

# activity_mapping = {'Walking': 'Jogging'}
# activity_mapping = {'Upstairs': 'Downstairs'}
# activity_mapping = {'Sitting': 'Standing'}

# activity_mapping = {'Jogging': 'Standing'}
# activity_mapping = {'Upstairs': 'Standing'}

activity_mapping = {}

def merging_activities(activity):
    if activity in activity_mapping:
        return activity_mapping[activity]
    else:
        return activity
    
features_of_all_activities_temp = features_of_all_activities.copy(True) 
features_of_all_activities_temp['activity'] = features_of_all_activities_temp['activity'].map(merging_activities) 
features_of_all_activities_temp



# In[33]:

# activity_of_interest = activities[2] 
# activity_of_interest = 'Standing'

# user_of_interest = user_ids[0]

# number_of_instances_per_bag = 12 * 1 
noise_level_pos = 0.5 #0 => supervised learning
noise_level_neg = 0.0 #0 => clean negative bags


#initialize variables
user_ids_temp = features_of_all_activities_temp['user'].value_counts().index.values.astype('int').tolist()
activities_temp = features_of_all_activities_temp['activity'].value_counts().index.tolist()

prediction_results = []


for noise_level_pos in np.arange(0.0, 1.0, 0.1):

# for noise_level_neg in np.arange(0.0, 1.0, 0.1):
    
    
        for activity_of_interest in activities_temp:

            for user_of_interest in user_ids_temp:

                positive_bags, negative_bags = get_bags_for_user(features_of_all_activities_temp, activity_of_interest, [user_of_interest], number_of_instances_per_bag, noise_level_pos, noise_level_neg)    

                if len(positive_bags) < 2 or len(negative_bags) < 2:
                    continue

                for algorithm, classifier in mil_classifiers.items():

                    print noise_level_pos, noise_level_neg, user_of_interest, algorithm

                    bags_prediction_df, instances_prediction_df = cross_val_score_MIL(classifier, positive_bags, negative_bags, cv_n_folds=2, shuffle = True)

                    prediction_results.append({
                        'activity': activity_of_interest,
                        'user': user_of_interest,
                        'noise_level_pos': noise_level_pos,
                        'noise_level_neg': noise_level_neg,
                        'algorithm': algorithm,
                        'bags_prediction_df': bags_prediction_df,
                        'instances_prediction_df': instances_prediction_df
                    }) 
    
prediction_results_df = pd.DataFrame(prediction_results)

HTML(alert_when_done)


# In[36]:

#select only jogging
# prediction_results_df_temp = filter_df(prediction_results_df, {'activity': 'Standing'}) 

prediction_results_df_temp = prediction_results_df

groupby = prediction_results_df_temp.groupby(['algorithm', 'noise_level_pos'])
# groupby = prediction_results_df_temp.groupby(['algorithm', 'noise_level_neg'])
# groupby = prediction_results_df.groupby(['algorithm', 'noise_level', 'activity']) #in case we want to look at the activities separately

groupby = groupby.agg({
    'bags_prediction_df': lambda x: pd.concat(x.values.tolist()),
    'instances_prediction_df': lambda x: pd.concat(x.values.tolist())
})

groupby = groupby.reset_index()
groupby


# In[37]:


# prediction_results_temp = filter_df(prediction_results_df, {'user': user_ids[4]})

# prediction_results_temp = filter_df(groupby, {'activity': activities[4]})

prediction_results_temp = groupby


prediction_results_temp['bags_precision'] = prediction_results_temp['bags_prediction_df'].map(lambda df: compute_precision_for_label(df, 1))
prediction_results_temp['bags_recall'] = prediction_results_temp['bags_prediction_df'].map(lambda df: compute_recall_for_label(df, 1))
prediction_results_temp['instances_precision'] = prediction_results_temp['instances_prediction_df'].map(lambda df: compute_precision_for_label(df, 1))
prediction_results_temp['instances_recall'] = prediction_results_temp['instances_prediction_df'].map(lambda df: compute_recall_for_label(df, 1))

prediction_results_temp['bags_f1'] = 2 * prediction_results_temp['bags_precision'] * prediction_results_temp['bags_recall']  / (prediction_results_temp['bags_precision'] + prediction_results_temp['bags_recall'])
prediction_results_temp['instances_f1'] = 2 * prediction_results_temp['instances_precision'] * prediction_results_temp['instances_recall']  / (prediction_results_temp['instances_precision'] + prediction_results_temp['instances_recall'])


prediction_results_temp.index = prediction_results_temp['noise_level_pos'] 



# In[38]:

algorithms = prediction_results_temp['algorithm'].value_counts().index.values
algorithms


# In[39]:

for algorithm in algorithms:
    filter_df(prediction_results_temp, {'algorithm': algorithm})[['bags_f1', 'instances_f1']].plot(ylim=(0,1))


##### Results of N-Fold CV

# In[40]:

# algorithms = ['SIL', 'MISVM', 'miSVM']

temp = pd.DataFrame()
for algorithm in algorithms:
#     temp[algorithm] = filter_df(prediction_results_temp, {'algorithm': algorithm})['instances_f1']
    temp[algorithm] = filter_df(prediction_results_temp, {'algorithm': algorithm})['bags_f1']
    
temp.plot(linewidth=4.0, ylim=(0,1))

xlabel('noise level')
ylabel('F1-score')
matplotlib.rc('font', **{'family' : 'sans-serif','weight' : 'normal', 'size'   : 15})
legend(prop={'size':15}, loc='center left', bbox_to_anchor=(1, 0.5))

# HTML(alert_when_done)


##### MIL Leave-one-person-out-validation

# In[ ]:

#VERY SLOW

def cross_val_leave_one_person_out_MIL(classifier):

    bags_prediction_df = pd.DataFrame()
    instances_prediction_df = pd.DataFrame()
    
    
    for i in range(len(user_ids)):

        user_of_interest = user_ids[i]
        
        not_user_of_interest = list(user_ids)
        not_user_of_interest.remove(user_ids[i])
        

        bags_positive_train, bags_negative_train = get_bags_for_user(features_of_all_activities, activity_of_interest, not_user_of_interest, number_of_instances_per_bag, noise_level)
        bags_positive_test, bags_negative_test = get_bags_for_user(features_of_all_activities, activity_of_interest, [user_of_interest], number_of_instances_per_bag, noise_level)

        
        bag_prediction_df, instance_prediction_df = learn_and_predict_MIL(classifier, bags_positive_train, bags_negative_train, bags_positive_test, bags_negative_test)
        
        
        bags_prediction_df = pd.concat([bags_prediction_df, bag_prediction_df])
        instances_prediction_df = pd.concat([instances_prediction_df, instance_prediction_df])
        
        
    return bags_prediction_df, instances_prediction_df
    
bags_prediction_df, instances_prediction_df = cross_val_leave_one_person_out_MIL(classifier)



## PLOT for PAPER

# In[ ]:

activities


# In[ ]:

jogging = filter_df(data_df, {'activity': 'Jogging'})[['x', 'y', 'z']]
walking = filter_df(data_df, {'activity': 'Walking'})[['x', 'y', 'z']]
downstairs = filter_df(data_df, {'activity': 'Downstairs'})[['x', 'y', 'z']]
upstairs = filter_df(data_df, {'activity': 'Upstairs'})[['x', 'y', 'z']]
sitting = filter_df(data_df, {'activity': 'Sitting'})[['x', 'y', 'z']]
standing = filter_df(data_df, {'activity': 'Standing '})[['x', 'y', 'z']]

# jogging.plot()


### Pattern localization

# In[ ]:

running = [walking.iloc[0:70], jogging.iloc[0:30], walking.iloc[0:100], standing.iloc[0:30], jogging.iloc[0:30], upstairs.iloc[0:70]]
not_running = [standing.iloc[0:100], downstairs.iloc[0:50], walking.iloc[0:100], standing.iloc[0:30], upstairs.iloc[0:50]]

running = [walking.iloc[0:20], jogging.iloc[0:100], standing.iloc[0:30], walking.iloc[0:50], jogging.iloc[0:0], upstairs.iloc[0:70]]
not_running = [walking.iloc[0:50], standing.iloc[0:20], walking.iloc[0:50], downstairs.iloc[0:40], standing.iloc[0:30],walking.iloc[0:50],  upstairs.iloc[0:50]]



activity_to_plot = not_running
# activity_to_plot = running

combined_activity = pd.concat(activity_to_plot)
combined_activity = combined_activity.reset_index()[['x', 'y', 'z']]
combined_activity.plot(figsize=(2,2), legend=False)

plt.xticks([])
plt.yticks([])
# legend(prop={'size':15}, loc='center left', bbox_to_anchor=(1, 0.5))
# xlabel('Time (s)')
# ylabel('Acceleration')


# In[ ]:

np.amax(combined_activity.values)


### Learning input

# In[ ]:

#Running

#Not running


## High-level Activities

# In[ ]:

opportunity_data_dir = '/Users/ntle/_workspaces/workspace27DeviceFree/_data/_activity_data/Opportunity/OpportunityUCIDataset/dataset/'
label_legend_src = opportunity_data_dir + 'label_legend.txt'

user_data_src = opportunity_data_dir + 'S1-ADL1.dat'
# user_data_src = opportunity_data_dir + 'S2-ADL5.dat'|

# user_data_src = opportunity_data_dir + 'S1-Drill.dat'


# In[ ]:

user_data_df.shape


# In[ ]:

label_legend_df = pd.read_csv(label_legend_src, sep='   -   ', index_col=0)
# label_legend_df


# In[ ]:

user_data_df = pd.read_csv(user_data_src, sep=' ', names = range(1, 251), index_col=0)


# In[ ]:

plot_labels_over_time(user_data_df[245]) #HL_Activity


plot_labels_over_time(user_data_df[244]) #Locomotion

plot_labels_over_time(user_data_df[246]) #LL_Left_Arm
plot_labels_over_time(user_data_df[247]) #LL_Left_Arm_Object

plot_labels_over_time(user_data_df[248]) #LL_Right_Arm
plot_labels_over_time(user_data_df[249]) #LL_Right_Arm_Object

plot_labels_over_time(user_data_df[250]) #ML_Both_Arms


## Recognizing unseen activities

# In[7]:

nuactiv_data_src = '../../_data/NuActiv/CMU_exercise_activity_dataset.csv'


# In[131]:

# Armband (Nexus S 4G Phone)
# Linear Acceleration features 
# Mean (x,y,z): 1-3
# Std (x,y,z): 4-6
# Pairwise correlation (xy, xz, yz): 7-9
# Local slope (x, y, z): 10-12
# Zero-crossing rate (x, y, z): 13-15

# Gravity Vector features
# Mean (x,y,z): 1-3
# Std (x,y,z): 4-6
# Pairwise correlation (xy, xz, yz): 7-9
# Local slope (x, y, z): 10-12
# Zero-crossing rate (x, y, z): 13-15

# Gyroscope features
# Mean (x,y,z): 1-3
# Std (x,y,z): 4-6
# Pairwise correlation (xy, xz, yz): 7-9
# Local slope (x, y, z): 10-12
# Zero-crossing rate (x, y, z): 13-15


# Wristwatch (MotoACTV)
# Linear Acceleration features
# Mean (x,y,z): 1-3
# Std (x,y,z): 4-6
# Pairwise correlation (xy, xz, yz): 7-9
# Local slope (x, y, z): 10-12
# Zero-crossing rate (x, y, z): 13-15

# Gravity Vector features
# Mean (x,y,z): 1-3
# Std (x,y,z): 4-6
# Pairwise correlation (xy, xz, yz): 7-9
# Local slope (x, y, z): 10-12
# Zero-crossing rate (x, y, z): 13-15


nuactiv_data_df = pd.read_csv(nuactiv_data_src, names = ['user', 'activity'] + range(75))

user_ids = nuactiv_data_df['user'].value_counts().index.values

nuactiv_data_df


### Data Exploration

# In[9]:

nuactiv_data_df['activity'].plot()


### Supervised learning

# In[10]:

classifiers = {}


# classifiers['GaussianNB'] = GaussianNB()

# classifiers['DecisionTreeClassifier'] = tree.DecisionTreeClassifier()

# classifiers['RandomForestClassifier'] = RandomForestClassifier(n_estimators=5)
# classifiers['ExtraTreesClassifier'] = ExtraTreesClassifier(n_estimators=5)


# classifiers['NearestCentroid'] = NearestCentroid()
# classifiers['KNeighborsClassifier'] = KNeighborsClassifier(n_neighbors=5)

# classifiers['LogisticRegression'] = LogisticRegression()
# classifiers['LinearSVC'] = svm.LinearSVC()
classifiers['SVC'] = svm.SVC(kernel='poly', max_iter=20000)

classifiers


# In[11]:

#n-fold cross validation
features_of_all_activities = nuactiv_data_df

user_ids = features_of_all_activities['user'].value_counts().index.values

# features_and_labels = filter_df(features_of_all_activities, {'user': user_ids[2]})
features_and_labels = features_of_all_activities

features, labels = get_features_and_labels_for_scikitlearn(features_and_labels)

#very good results
for algorithm, classifier in classifiers.items():
    classification_results = cross_validation.cross_val_score(classifier, features, labels, cv=10)
    print classifier.__class__.__name__, '\t & \t', classification_results.mean().round(2), '\t & \t', classification_results.std().round(2), ' \\\\'



# In[12]:

#leave-one-user-out cross validation

accuracies = {}
for algorithm, classifier in classifiers.items():

    accuracy_per_classifier = []
    #cross-validation
    for i in range(len(user_ids)):

        testing_user = user_ids[i] 

        testing_features_and_labels, training_features_and_labels = filter_df(features_of_all_activities, {'user': testing_user}, return_complement=True)

        
        #switch 2 axes
#         testing_features = swap_axes(testing_features, 'x', 'y')
#         testing_features = swap_axes(testing_features, 'z', 'y')
#         testing_features = swap_axes(testing_features, 'x', 'y')
        
        training_features, training_labels = get_features_and_labels_for_scikitlearn(training_features_and_labels)
        testing_features, testing_labels = get_features_and_labels_for_scikitlearn(testing_features_and_labels)
        
        classifier.fit(training_features, training_labels)
        predictions = classifier.predict(testing_features)
        
        accuracy = np.mean(testing_labels == predictions)
        accuracy_per_classifier.append(accuracy)
        
        print i, accuracy
#         if accuracy == nan:
#             print predictions
        
        
    accuracies[algorithm] = accuracy_per_classifier

    
    
    
for algorithm, accuracy in accuracies.items():
    print '\n%s Accuracy: %.2f%% (%.2f) (%s) ' % (algorithm, np.average(accuracy), np.std(accuracy), accuracy)


### Attribute detector

# In[14]:

#activity-attribute matrix

features_of_all_activities = nuactiv_data_df


activities = ['Bench Dips', 'Squat Upright Row', 'DB Side Raises', 'DB Shoulder Press', 'Dumbbell Curl', 'Triceps Extension', 'Chest Press', 'Push Up', 'Dumbbell Fly', 'Bent−Over Row']
attributes = ['ArmUp', 'ArmDown', 'ArmFwd', 'ArmBack', 'ArmSide', 'ArmCurl', 'SquatStand']

matrix = [
    [0, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 1, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0 ,0, 1, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 1, 0],
    [0, 1 ,0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 1]
]

activity_attribute_df = pd.DataFrame(matrix, index=activities, columns = attributes)
activity_attribute_df



#TODO: find these attributes and attributes values automatically from the data (attributes = bases, attribute values = contribution)


# In[15]:

def get_positive_activies_ids(activity_attribute_df, attribute, return_complement=True):
    
    #identify positive and negative instances
    positive_activities = activity_attribute_df[activity_attribute_df[attribute] == 1].index
    negative_activities = activity_attribute_df[activity_attribute_df[attribute] != 1].index

    positive_activities_ids = [activities.index(x) for x in positive_activities]
    negative_activities_ids = [activities.index(x) for x in negative_activities]

    if return_complement:
        return positive_activities_ids, negative_activities_ids
    else:
        return positive_activities_ids
    

def get_features_and_labels(features_of_all_activities, activity_attribute_df, attribute):


    positive_activities_ids, negative_activities_ids = get_positive_activies_ids(activity_attribute_df, attribute, return_complement=True)

    #labeling features with positive and negative labels
    features_and_labels_pos, features_and_labels_neg = filter_df(features_of_all_activities, {'activity': positive_activities_ids}, operation='in', return_complement=True)

    features_and_labels_pos['activity'] = 1 
    features_and_labels_neg['activity'] = -1
    # len(features_and_labels_pos), len(features_and_labels_neg)

    features_and_labels = pd.concat([features_and_labels_pos, features_and_labels_neg])
    return features_and_labels
    
    
    
attribute = attributes[1]
features_and_labels = get_features_and_labels(features_of_all_activities, activity_attribute_df, attribute)
features_and_labels


# In[16]:

#n-fold cross validation
def n_fold_cv(classifiers, features_and_labels, n_fold=10):
    
    features, labels = get_features_and_labels_for_scikitlearn(features_and_labels)
    for algorithm, classifier in classifiers.items():
        classification_results = cross_validation.cross_val_score(classifier, features, labels, cv=n_fold)
        print classifier.__class__.__name__, '\t & \t', classification_results.mean().round(2), '\t & \t', classification_results.std().round(2), ' \\\\'

        
n_fold_cv(classifiers, features_and_labels)    


# In[79]:

#supervised learning
results_df = pd.DataFrame()
for attribute in attributes:

    print attribute
    
    features_and_labels = get_features_and_labels(features_of_all_activities, activity_attribute_df, attribute)
    result_df = leave_one_person_out_cv(classifiers, features_and_labels)
    result_df['attribute'] = attribute
    results_df = pd.concat([results_df, result_df])
    


# In[18]:

groupby = results_df.groupby(['algorithm', 'attribute'])
prediction_results_df = groupby.apply(computing_result_metrics)
prediction_results_df


# In[19]:

prediction_results_df.plot()


### MIL Attribute detector

# In[25]:

#merge instances into bags

#bag = one activity of one person => for each activity we 20 bags (20 people)

def get_bags(df): 
    features, labels = get_features_and_labels_for_scikitlearn(df)
    return features.values

features_of_all_activities = nuactiv_data_df
groupby = features_of_all_activities.groupby(['user', 'activity'])

bags = groupby.apply(get_bags)
bags = bags.reset_index()

bags


# In[23]:

#generate positive and negative bags
attribute = attributes[1]

features_and_labels = get_features_and_labels(bags, activity_attribute_df, attribute)
features_and_labels


# In[91]:

#MIL
 
#   SVM     : a standard supervised SVM
#   SIL     : trains a standard SVM classifier after applying bag labels to each
#             instance
#   MISVM   : the MI-SVM algorithm of Andrews, Tsochantaridis, & Hofmann (2002)
#   miSVM   : the mi-SVM algorithm of Andrews, Tsochantaridis, & Hofmann (2002)

#   NSK     : the normalized set kernel of Gaertner, et al. (2002)
#   STK     : the statistics kernel of Gaertner, et al. (2002)

#   MissSVM : the semi-supervised learning approach of Zhou & Xu (2007)
#   MICA    : the MI classification algorithm of Mangasarian & Wild (2008)

#   sMIL    : sparse MIL (Bunescu & Mooney, 2007)
#   stMIL   : sparse, transductive  MIL (Bunescu & Mooney, 2007)
#   sbMIL   : sparse, balanced MIL (Bunescu & Mooney, 2007)


# Construct classifiers
verbose = False
kernel = 'linear'
C = 1.0
max_iters = 50
# sv_cutoff = 1e-7
sv_cutoff = 0.1

mil_classifiers = {}

mil_classifiers['SIL'] = misvm.SIL(kernel=kernel, C=C, verbose=verbose)

# mil_classifiers['miSVM'] = misvm.miSVM(kernel=kernel, C=C, max_iters=max_iters, verbose=verbose)
# mil_classifiers['MISVM'] = misvm.MISVM(kernel=kernel, C=C, max_iters=max_iters, verbose=verbose)

# mil_classifiers['MICA'] = misvm.MICA(kernel=kernel, C=C, verbose=verbose)
# mil_classifiers['MissSVM'] = misvm.MissSVM(kernel=kernel, C=C, verbose=verbose)

# mil_classifiers['sMIL'] = misvm.sMIL(kernel=kernel, C=C, verbose=verbose)
# mil_classifiers['stMIL'] = misvm.stMIL(kernel=kernel, C=C, verbose=verbose)
# mil_classifiers['sbMIL'] = misvm.sbMIL(kernel=kernel, C=C, verbose=verbose)

# mil_classifiers['NSK'] = misvm.NSK(kernel=kernel, C=C, verbose=verbose)
# mil_classifiers['STK'] = misvm.STK(kernel=kernel, C=C, verbose=verbose)



# In[92]:

#MIL learning

results_df = pd.DataFrame()
for attribute in attributes:

    time_start()
    print attribute
    
    features_and_labels = get_features_and_labels(bags, activity_attribute_df, attribute)
    
    result_df = leave_one_person_out_cv(mil_classifiers, features_and_labels)
    result_df['attribute'] = attribute
    results_df = pd.concat([results_df, result_df])
    

    time_stop()



# In[85]:

groupby_attributes = ['algorithm', 'attribute', 'user']

results_metrics_df = result_df.groupby(groupby_attributes).apply(computing_result_metrics)
results_metrics_df


### Converting data to Weka Format

# In[132]:

attribute = attributes[0]

nuactiv_data_to_export_df = nuactiv_data_df

nuactiv_data_to_export_df['bag_id'] = nuactiv_data_to_export_df['user'].astype('string') + '_' + nuactiv_data_to_export_df['activity'].astype('string')
nuactiv_data_to_export_df['bag_id']

nuactiv_data_to_export_df['class'] = get_features_and_labels(nuactiv_data_to_export_df, activity_attribute_df, attribute)['activity'] == 1
nuactiv_data_to_export_df['class'].value_counts()


# In[138]:

user = user_ids[0]

testing, training = filter_df(nuactiv_data_to_export_df, {'user': user}, return_complement=True)


# In[136]:

testing


# In[139]:

training.ix[:,['bag_id'] + range(75) + ['class']].to_csv('/home/le/mil_weka_training.csv', index=False)
testing.ix[:,['bag_id'] + range(75) + ['class']].to_csv('/home/le/mil_weka_testing.csv', index=False)


# In[41]:




## Multi-Label Learning (MLL)

# In[103]:

from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
iris = datasets.load_iris()
X, y = iris.data, iris.target
OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)


## TODO

                TODO: Find suitable dataset - with a large number of activities 


TODO: Composite activities - cutting + 


TODO: 


TODO: Why does bag accuracy not decrease with the noise level?
- VALIDATE THIS - intuition - high false positives rate - when only just one element is falsely predicted as positive, the whole bag is falsely predicted as positive
    - if activity of interest is walking, just one 




    
TODO: weighted precision - normalize the results based on the number of bags in each bag
    
TODO: why MISVM fails at 80% noise level for walking? 
    






DONE: compute this for all activities and all users, not only one activity or one user

DONE: why is Bag accuracy lower than instance accuracy? => should it not be the other way around?
- accuracy is not a good metric - precision and recall is better since is focuses more on positive instances in the binary classification


    
DONE: Why is MISVM better than miSVM with increase noise level?
- background
    - mi-SVM - all instances in positive bags matter
    - MI-SVM - only one instance per positive bags matters
- noise level
    - mi-SVM performs similarly to the supervised approach when there is no noise
    - mi-SVM performs better than MI-SVM when the noise level is low
    - increasing noise level does not effect the performance of MI-SVM since it considers only the "most positive instance" 
    


DONE: What does this mean for the pattern localization? - can we visualize this?
- instance F1 score corresponds to pattern localization accuracy

  
  


    

                
# In[ ]:



