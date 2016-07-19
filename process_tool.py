
import numpy as np
import matplotlib.pyplot as plt
import csv as csv 
from sklearn import svm
import warnings 
from argparse import ArgumentParser
from path import Path
from collections import defaultdict
import pandas as pd
import os,glob
from pandas import *
from math import *
from scipy.fftpack import fft
from numpy import mean, sqrt, square
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
def compute_power_spectrum(series, frequency = 100):

    ps = np.abs(np.fft.fft(series))**2
    freqs = np.fft.fftfreq(series.size, 1/frequency)
    idx = np.argsort(freqs)
    idx = idx[len(idx)/2:]

#     plt.plot(freqs[idx], ps[idx])

    return ps[len(ps)/2:]


def get_columns(df, except_columns):
    return [col for col in df.columns if col not in except_columns]

def filter_df(df, key_value_pairs, operation='equal', return_complement=False):
    for key in key_value_pairs:
        print('start filter:',key)
        if operation=='equal':
            selection = df[key] == key_value_pairs[key]
            
        elif operation=='not_equal':
            selection = df[key] != key_value_pairs[key]
            
        elif operation=='in':
            selection = df[key].isin(key_value_pairs[key])
            
        elif operation=='not_in':
            selection = df[key].notin(key_value_pairs[key])
    
    selected = df[selection]
    print('filter complete:',selected)
    if not return_complement:
        return selected
    else:
        complement = df[selection==False]
        return selected, complement
        
def get_items_from_list(l, indexes):
    items = itemgetter(*indexes)(l)
    
    if type(items) == tuple:
        items = list(items)
    else:
        items = [items]
    
    return items

def get_features_and_labels_for_scikitlearn(df):
    df = df.dropna()
    
    feature_columns = get_columns(df, ['User','activity'])
    features = df.ix[:,feature_columns] #make sure to select columns
    labels = df['activity']
    return features, labels
def leave_one_person_out_cv(classifiers, features_and_labels):
    results = []
    
    user_ids = features_and_labels['User'].value_counts().index.values
    
    for i in range(len(user_ids)):

        
        testing_user = user_ids[i] 
        print ("testing_user: ", testing_user)
        
        testing_features_and_labels, training_features_and_labels = filter_df(features_and_labels, {'user': testing_user}, return_complement=True)
        
        training_features, training_labels = get_features_and_labels_for_scikitlearn(training_features_and_labels)
        testing_features, testing_labels = get_features_and_labels_for_scikitlearn(testing_features_and_labels)
        
        
        if training_features.shape[1] == 1: #handle MIL bags instead of traditional features
            training_features = [x[0] for x in training_features.values]
            testing_features = [x[0] for x in testing_features.values]

        
        for algorithm, classifier in classifiers.items(): 
            
            print (algorithm)
            
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
    