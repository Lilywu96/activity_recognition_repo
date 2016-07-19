# import csv
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
from process_tool import *
from Feature_Extractor import *
from fft import *
from dataparser import *
from dataparsertest import *
warnings.filterwarnings("ignore")
#%matplotlib inline
def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument("--dataset_path", default="/Users/LilyWU/Documents/activity_recognition_repo/PAMAP2_Dataset/Protocol")
    argparser.add_argument("--file_path", default="/Users/LilyWU/Documents/activity_recognition_repo/Activity recognition exp 2/Watch_accelerometer_new.csv")
    argparser.add_argument("--save_path",default="/Users/LilyWU/Documents/activity_recognition_repo/Activity recognition exp 2/save")
    # argparser.add_argument("--save_path", default="patches/")
    return argparser.parse_args()

if __name__ == '__main__':
    dataset_path="/Users/LilyWU/Documents/PAMAP/PAMAP2_Dataset/"
    parser=performDataParsing(dataset_path)
    sessions=parser.Parsing()

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

    





 