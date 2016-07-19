#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn import svm
from numpy import genfromtxt
from sklearn.metrics import confusion_matrix

def sweep(train_file, test_file, C = 8000, delimiter=','):
  train = genfromtxt(train_file,delimiter=delimiter)
  test = genfromtxt(test_file,delimiter=delimiter)
  train_label = train[:,0]
  train_data = train[:,1:]
  test_label = test[:,0]
  test_data = test[:,1:]
  lst = range(2000,3000,100)
  for C in lst:
    print "C = {0}".format(C)
    svc = svm.LinearSVC(C=C)
    svc.fit(train_data, train_label)
    pred = svc.predict(test_data)
    print svc.score(test_data, test_label)
    logreg = linear_model.LogisticRegression(C=C)
    logreg.fit(train_data,train_label)
    pred = logreg.predict(test_data)
    print logreg.score(test_data,test_label)


def classify(train_file, test_file, C = 8000, delimiter=','):
  train = genfromtxt(train_file,delimiter=delimiter)
  test = genfromtxt(test_file,delimiter=delimiter)
  train_label = train[:,0]
  train_data = train[:,1:]
  test_label = test[:,0]
  test_data = test[:,1:]
  #lst = range(300,2000,100)
  print "C = {0}".format(C)
  #svc = svm.LinearSVC(C=C)
  #svc.fit(train_data, train_label)
  #pred = svc.predict(test_data)
  #print svc.score(test_data, test_label)
  #con = confusion_matrix(test_label, pred)
  logreg = linear_model.LogisticRegression(C=C)
  logreg.fit(train_data,train_label)
  pred = logreg.predict(test_data)
  prob = logreg.predict_proba(test_data)
  return_prob = []
  current_label = None
  prob_list = None
  count = 0
  for i in range(0,test_label.shape[0]):
    if current_label == None:
      current_label = test_label[i]
      prob_list = prob[i]
      count = 1
    elif current_label == test_label[i]:
      prob_list += prob[i]
      count += 1
    else:
      prob_list = prob_list/count
      return_prob.append(prob_list)
      current_label = test_label[i]
      prob_list = prob[i]
      count = 1
  prob_list = prob_list/count
  return_prob.append(prob_list)
  print logreg.score(test_data,test_label)
  return (logreg, return_prob)
  #con = confusion_matrix(test_label, pred)
  


def classify_local(train_file, test_file, C = 8000, delimiter=','):
  train = genfromtxt(train_file,delimiter=delimiter)
  test = genfromtxt(test_file,delimiter=delimiter)
  train_label = train[:,0]
  train_data = train[:,1:]
  test_label = test[:,0]
  test_row = test[:,1]
  test_data = test[:,2:]
  #lst = range(300,2000,100)
  print "C = {0}".format(C)
  #svc = svm.LinearSVC(C=C)
  #svc.fit(train_data, train_label)
  #pred = svc.predict(test_data)
  #print svc.score(test_data, test_label)
  #con = confusion_matrix(test_label, pred)
  logreg = linear_model.LogisticRegression(C=C)
  logreg.fit(train_data,train_label)
  pred = logreg.predict(test_data)
  prob = logreg.predict_proba(test_data)
  return_prob = {}
  current_label = None
  prob_list = []
  for i in range(0,test_label.shape[0]):
    print i
    if current_label == None:
      current_label = test_label[i]
      prob_list.append(prob[i])
    elif current_label == test_label[i]:
      prob_list.append(prob[i])
    else:
      prob_list = np.array(prob_list)
      return_prob[test_row[i]] = (prob_list.max(axis=0), prob_list.argmax(axis=0))
      current_label = test_label[i]
      prob_list = [prob[i]]
  prob_list = np.array(prob_list)
  return_prob[test_row[i]] = (prob_list.max(axis=0), prob_list.argmax(axis=0))
  print logreg.score(test_data,test_label)
  return (logreg, return_prob)
  #con = confusion_matrix(test_label, pred)


#(logreg_g, prob_g) = classify('./outputs/huyun_activity_primitives_global_train16.csv', './outputs/huyun_activity_primitives_global_test16.csv', 8000)
(logreg_g, prob_g) = classify('./outputs/h_train_glob_final', './outputs/h_test_glob_final', 3000)
(logreg_l, prob_l) = classify_local('./outputs/h_train_tf_final', './outputs/h_test_tf_marker', 2600)
#print len(prob_l), len(prob_g)
test = genfromtxt('./outputs/h_test_glob_final',delimiter=',')
test_label = test[:,0]
test_data = test[:,1:]
C = map(lambda(x):0.1*x,range(1,10))
for c in C:
  print c
  correct = 0
  for i in range(0,len(test_label)):
    if i in prob_l:
      check = c*prob_l[i][0] + (1-c)*prob_g[i]
      predict = logreg_g.classes_[int(check.argmax())]
      actual = int(test_label[i])
    else:
      check = prob_g[i]
      predict = logreg_g.classes_[int(check.argmax())]
      actual = int(test_label[i])
    if predict == actual:
      correct += 1
  print correct/float(len(test_label))

#print test_label
#print lg
#print prob_g.argmax(axis=1)
#print prob_l[1][1]
#print prob_g[1].argmax()
#print prob_g[1:10].argmax(axis=0)
#print prob_g.argmax(axis=1) - lg
#print logreg_g.classes_
#print logreg_g.classes_[logreg_g.decision_function(test_data).argmax(axis=1)]
#sweep('./outputs/h_train_glob_final', './outputs/h_test_glob_final', 8000)
#sweep('./outputs/h_train_tf_final', './outputs/h_test_tf_final', 8000)
#print len(prob_g)
#(logreg_l, prob_l) = classify('./outputs/huyun_activity_primitives_tf_train16.csv', './outputs/huyun_activity_primitives_tf_test16.csv', 22000)


#for i in range(0,con.shape[0]):
#    s = sum(con[i,:])
#    print con[i,i]/float(s)


