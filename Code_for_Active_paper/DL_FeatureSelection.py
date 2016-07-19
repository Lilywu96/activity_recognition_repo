#!/usr/local/bin/python

import os, sys
import string
import linecache

import numpy as np
import pandas as pd
from scipy.cluster.vq import whiten

def isSameLabel(df):
  df_matrix = df.values
  
  l = df_matrix[0]
  for i in df_matrix:
    if ( i != l ):
      return False
  return True

def normalize(data):
  features = np.array(data)
  print features 
  feature_whiten = whiten(features) 
  return feature_whiten

def SelectFeature2File(infile, outfile):
  slide_window = 64
  shift = int(slide_window * 0.5)
  data = pd.read_csv(infile, sep = ' ', header = None)
  output_feature = []
  output_label = []

  df1 = data[data.columns[1:4]]
  df2 = data[data.columns[0]]

  n = len(df1[1])
  start = 0
  end = start + slide_window
  print end
  print n
  while ( end <= n ):
    feature_slice = df1[start:end]
    label_slice = df2[start:end]
    
    if ( isSameLabel(label_slice) ):
      output_feature.append((feature_slice.values).flatten())
      output_label.append(label_slice.values[0])
    start = end - shift
    end = start + slide_window
    print end

  feature_whiten = normalize(output_feature)
  
  i = 0
  myfile = open(outfile, 'w')
  for row in feature_whiten:
    l = output_label[i]
    if l == 32:
      l = -1
    elif l == 48:
      l = 0
    elif l == 49:
      l = 1
    elif l == 50:
      l = 2
    elif l == 51:
      l = 3
    elif l == 52:
      l = 4
    elif l == 53:
      l = 5
    elif l == 54:
      l = 6
    elif l == 55:
      l = 7
    elif l == 56:
      l = 8
    elif l == 57:
      l = 9
    if ( l >= 0 ):
      line = ""
      x = ""
      y = ""
      z = ""
      axis = 0
      for f in row.tolist():
        if ( axis % 3 == 0 ):
          x += str(f) + " "
        elif ( axis % 3 == 1 ):
          y += str(f) + " "
        elif ( axis % 3 == 2):
          z += str(f) + " "
          axis += 1
      line = x + y + z + str(l) + '\n'   
#    line += str(f) + " "
#    line += str(l) + '\n'
      myfile.write(line)
    i += 1
 # df1 = pd.DataFrame(feature_whiten)
 # df2 = pd.DataFrame(output_label)
 # df = pd.merge(df1, df2, on=df1.index, how = 'outer')
 # dff = df[df.columns[1:194]] 
 # dff.to_csv(outfile, header = None, index = False, sep = ' ')


  myfile.close()

if __name__ == "__main__":

  filename = "sensor29.txt"
  input_file = "/Users/zengming/Developer/dataset/SkodaMiniCP/" + filename
  output_file = "/Users/zengming/Developer/dataset/SkodaMiniCP/preprocessing/data/" + filename + "_64interval"
  
  SelectFeature2File(input_file, output_file)
