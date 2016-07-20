# import csv
import numpy as np
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
from process_tool import *
from Feature_Extractor import *
from dataparser import *
from dataparsertest import *
from load_PAMAP2 import Loading_PAMAP2
from load_HAPT import Loading_HAPT
warnings.filterwarnings("ignore")
#%matplotlib inline
# def parse_args():
#     argparser = ArgumentParser()
#     argparser.add_argument("--HAPT_path", default="/Users/LilyWU/Documents/activity_recognition_repo/HAPT Data Set/RawData")
#     argparser.add_argument("--PAMAP2_path", default="/Users/LilyWU/Documents/activity_recognition_repo/PAMAP2_Dataset/Protocol")
#     argparser.add_argument("--",default="/Users/LilyWU/Documents/activity_recognition_repo/Activity recognition exp 2/save")
#     # argparser.add_argument("--save_path", default="patches/")
#     return argparser.parse_args()
HAPT_folder="HAPT Data Set/RawData"
PAMAP2_folder="PAMAP2_Dataset/Protocol"

def Loading(dataset,percentage):
   data={}
   data['activity']=list()
   data['timestamp']=list()
   data['x']=list()
   data['y']=list()
   data['z']=list()
   data['User']=list()

   if(dataset=="HAPT"):
      paths=glob.glob(HAPT_folder+'/*.txt')
      for filename in paths: 

          if(filename!=HAPT_folder+'/labels.txt'):
              Loading_HAPT(filename,data)
   if(dataset=="PAMAP2"):
      paths=glob.glob(PAMAP2_folder+'/*.txt') 
      id=1
      for filepath in paths: 
          Loading_PAMAP2(filepath,id,data)
          id=id+1
          # return piece
#def sliding_windowing():

if __name__ == '__main__':
    Loading('HAPT',100)
    


    
    # final_data=pd.DataFrame()
    # for f in sub_dir:
    #     data=pd.read_csv(f, names=['time','Ay','Az','label'],header=None)
    #     final_data=concat([final_data,data], ignore_index=True)
    # users_with_data_from_all_activities = data_df.groupby('User')['gt'].value_counts().unstack().dropna().index
    # data_df = data_df[data_df['User'].isin(users_with_data_from_all_activities)]
    # print (data_df)
    # #initialize variables
    # user_ids = data_df['User'].value_counts().index.values.tolist()
    # print('users:',user_ids)
    # activities = data_df['gt'].value_counts().index.values.tolist()
    # Model=data_df['Model'].value_counts().index.values.tolist()

    ##Baseline
    
#### Classification

    





 