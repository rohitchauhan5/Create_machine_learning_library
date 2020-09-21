# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 20:36:16 2020

@author: consultant138
"""
import os
os.chdir('D:\\ViteosModel')


import numpy as np
import pandas as pd
#from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from tqdm import tqdm
import pickle
import datetime as dt
import sys
from ViteosMongoDB import  ViteosMongoDB_Class as mngdb
from datetime import datetime,date,timedelta
from pandas.io.json import json_normalize
import dateutil.parser
from difflib import SequenceMatcher
import pprint
import json
from pandas import merge
import re

import dask.dataframe as dd
import glob
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from dateutil.parser import parse
import operator
import itertools
from sklearn.feature_extraction.text import CountVectorizer

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# In[2]:


cols = ['Currency','Account Type','Accounting Net Amount',
#'Accounting Net Amount Difference','Accounting Net Amount Difference Absolute ',
'Activity Code','Age','Age WK',
'Asset Type Category','Base Currency','Base Net Amount','Bloomberg_Yellow_Key',
'Cust Net Amount',
#'B-P Net Amount Difference','B-P Net Amount Difference Absolute',
'BreakID',
'Business Date','Cancel Amount','Cancel Flag','CUSIP','Custodian',
'Custodian Account',
'Derived Source','Description','Department','ExpiryDate','ExternalComment1','ExternalComment2',
'ExternalComment3','Fund','FX Rate','Interest Amount','InternalComment1','InternalComment2',
'InternalComment3','Investment Type','Is Combined Data','ISIN','Keys',
'Mapped Custodian Account','Net Amount Difference','Net Amount Difference Absolute','Non Trade Description',
'OTE Custodian Account',
#'Predicted Action','Predicted Status','Prediction Details',
'Price','Prime Broker',
'Quantity','SEDOL','Settle Date','SPM ID','Status','Strike Price',
'System Comments','Ticker','Trade Date','Trade Expenses','Transaction Category','Transaction ID','Transaction Type',
'Underlying Cusip','Underlying Investment ID','Underlying ISIN','Underlying Sedol','Underlying Ticker','Source Combination','_ID']
#'UnMapped']

add = ['ViewData.Side0_UniqueIds', 'ViewData.Side1_UniqueIds',
      # 'MetaData.0._RecordID','MetaData.1._RecordID',
       'ViewData.Task Business Date']


# In[3]:


new_cols = ['ViewData.' + x for x in cols] + add

common_cols = ['ViewData.Accounting Net Amount', 'ViewData.Age',
'ViewData.Age WK', 'ViewData.Asset Type Category',
'ViewData.B-P Net Amount', 'ViewData.Base Net Amount','ViewData.CUSIP', 
 'ViewData.Cancel Amount',
       'ViewData.Cancel Flag',
#'ViewData.Commission',
        'ViewData.Currency', 'ViewData.Custodian',
       'ViewData.Custodian Account',
       'ViewData.Description','ViewData.Department', 'ViewData.ExpiryDate', 'ViewData.Fund',
       'ViewData.ISIN',
       'ViewData.Investment Type',
      # 'ViewData.Keys',
       'ViewData.Mapped Custodian Account',
       'ViewData.Net Amount Difference',
       'ViewData.Net Amount Difference Absolute',
        #'ViewData.OTE Ticker',
        'ViewData.Price',
       'ViewData.Prime Broker', 'ViewData.Quantity',
       'ViewData.SEDOL', 'ViewData.SPM ID', 'ViewData.Settle Date',
       
  #  'ViewData.Strike Price',
               'Date',
       'ViewData.Ticker', 'ViewData.Trade Date',
       'ViewData.Transaction Category',
       'ViewData.Transaction Type', 'ViewData.Underlying Cusip',
       'ViewData.Underlying ISIN',
       'ViewData.Underlying Sedol','filter_key','ViewData.Status','ViewData.BreakID',
              'ViewData.Side0_UniqueIds','ViewData.Side1_UniqueIds','ViewData._ID']

model_cols = [
            'SideA.ViewData.B-P Net Amount', 
              #'SideA.ViewData.Cancel Flag', 
              #'SideA.new_desc_cat',
             # 'SideA.ViewData.Description',
             # 'SideA.ViewData.Department',
   
    
              
             # 'SideA.ViewData.Price',
             # 'SideA.ViewData.Quantity',
             #'SideA.ViewData.Investment Type', 
              #'SideA.ViewData.Asset Type Category', 
              'SideB.ViewData.Accounting Net Amount', 
              #'SideB.ViewData.Cancel Flag', 
             # 'SideB.ViewData.Description',
              # 'SideB.ViewData.Department',
              
             # 'SideB.ViewData.Price',
             # 'SideB.ViewData.Quantity',
             # 'SideB.new_desc_cat',
             # 'SideB.ViewData.Investment Type', 
              #'SideB.ViewData.Asset Type Category', 
              'Trade_Date_match', 'Settle_Date_match', 
                'Amount_diff_2', 
              'Trade_date_diff', 'Settle_date_diff', 'SideA.ISIN_NA', 'SideB.ISIN_NA', 
             # 'ViewData.Combined Fund',
              'ViewData.Combined Transaction Type', 'Combined_Desc','Combined_TType',
             # 'SideA.TType', 'SideB.TType', 
              'abs_amount_flag',
    'tt_map_flag', 
              'All_key_nan','new_key_match', 'new_pb1',
              'SideB.Date','SideA.ViewData.Settle Date','SideB.ViewData.Settle Date',
            'SideA.ViewData._ID', 'SideB.ViewData._ID','SideB.ViewData.Side0_UniqueIds', 'SideA.ViewData.Side1_UniqueIds',
              'SideB.ViewData.Status', 'SideB.ViewData.BreakID_B_side',
              'SideA.ViewData.Status', 'SideA.ViewData.BreakID_A_side'] 
              #'label']

model_cols_2 =[#'SideA.ViewData.B-P Net Amount', 
              #'SideA.ViewData.Cancel Flag', 
              #'SideA.new_desc_cat',
             # 'SideA.ViewData.Description',
             # 'SideA.ViewData.Department',
   
    
              
             # 'SideA.ViewData.Price',
             # 'SideA.ViewData.Quantity',
             #'SideA.ViewData.Investment Type', 
              #'SideA.ViewData.Asset Type Category', 
              #'SideB.ViewData.Accounting Net Amount', 
              #'SideB.ViewData.Cancel Flag', 
             # 'SideB.ViewData.Description',
              # 'SideB.ViewData.Department',
              
             # 'SideB.ViewData.Price',
             # 'SideB.ViewData.Quantity',
             # 'SideB.new_desc_cat',
             # 'SideB.ViewData.Investment Type', 
              #'SideB.ViewData.Asset Type Category', 
              'Trade_Date_match', 'Settle_Date_match', 
              #  'Amount_diff_2', 
              'Trade_date_diff', 'Settle_date_diff', 'SideA.ISIN_NA', 'SideB.ISIN_NA', 
             # 'ViewData.Combined Fund',
              'ViewData.Combined Transaction Type', 'Combined_Desc','Combined_TType',
             # 'SideA.TType', 'SideB.TType', 
              'abs_amount_flag',
    'tt_map_flag', 
              'All_key_nan','new_key_match', 'new_pb1',
              'SideB.Date','SideA.ViewData.Settle Date','SideB.ViewData.Settle Date',
            'SideA.ViewData._ID', 'SideB.ViewData._ID','SideB.ViewData.Side0_UniqueIds', 'SideA.ViewData.Side1_UniqueIds',
              'SideB.ViewData.Status', 'SideB.ViewData.BreakID_B_side',
              'SideA.ViewData.Status', 'SideA.ViewData.BreakID_A_side'] 
              #'label']

#### Closed break functions - Begin #### 

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def dictionary_exclude_keys(fun_dict, fun_keys_to_exclude):
    return {x: fun_dict[x] for x in fun_dict if x not in fun_keys_to_exclude}

def write_dict_at_top(fun_filename, fun_dict_to_add):
    with open(fun_filename, 'r+') as f:
        fun_existing_content = f.read()
        f.seek(0, 0)
        f.write(json.dumps(fun_dict_to_add, indent = 4))
        f.write('\n')
        f.write(fun_existing_content)

def normalize_bp_acct_col_names(fun_df):
    bp_acct_col_names_mapping_dict = {
                                      'ViewData.Cust Net Amount' : 'ViewData.B-P Net Amount',
                                      'ViewData.Cust Net Amount Difference' : 'ViewData.B-P Net Amount Difference',
                                      'ViewData.Cust Net Amount Difference Absolute' : 'ViewData.B-P Net Amount Difference Absolute',
                                      'ViewData.CP Net Amount' : 'ViewData.B-P Net Amount',
                                      'ViewData.CP Net Amount Difference' : 'ViewData.B-P Net Amount Difference',
                                      'ViewData.CP Net Amount Difference Absolute' : 'ViewData.B-P Net Amount Difference Absolute',
                                      'ViewData.PMSVendor Net Amount' : 'ViewData.Accounting Net Amount'
                                        }
    fun_df.rename(columns = bp_acct_col_names_mapping_dict, inplace = True)
    return(fun_df)



# M X M and N X N architecture for closed break prediction
def closed_cols():
    cols_for_closed_list = ['Status','Source Combination','Mapped Custodian Account',
                   'Accounting Currency','B-P Currency', 
                   'Transaction ID','Transaction Type','Description','Investment ID',
                   'Accounting Net Amount','B-P Net Amount', 
                   'InternalComment2','Custodian','Fund']
    cols_for_closed_list = ['ViewData.' + x for x in cols_for_closed_list]
    cols_for_closed_x_list = [x + '_x' for x in cols_for_closed_list] + ['ViewData.Side0_UniqueIds_x','ViewData.Side1_UniqueIds_x']
    cols_for_closed_y_list = [x + '_y' for x in cols_for_closed_list] + ['ViewData.Side0_UniqueIds_y','ViewData.Side1_UniqueIds_y']
    cols_for_closed_x_y_list = cols_for_closed_x_list + cols_for_closed_y_list
    return({
            'cols_for_closed' : cols_for_closed_list,
            'cols_for_closed_x' : cols_for_closed_x_list,
            'cols_for_closed_y' : cols_for_closed_y_list,
            'cols_for_closed_x_y' : cols_for_closed_x_y_list
            })

def cleaned_meo(#fun_filepath_meo, 
                fun_meo_df):
#    meo = pd.read_csv(fun_filepath_meo)           .drop_duplicates()           .reset_index()           .drop('index',1)
    meo = fun_meo_df
    meo = normalize_bp_acct_col_names(fun_df = meo)
    
    meo = meo[~meo['ViewData.Status'].isin(['SMT','HST', 'OC', 'CT', 'Archive','SMR'])] 
    meo = meo[~meo['ViewData.Status'].isnull()]           .reset_index()           .drop('index',1)
    
    meo['Date'] = pd.to_datetime(meo['ViewData.Task Business Date'])
    meo = meo[~meo['Date'].isnull()]           .reset_index()           .drop('index',1)
    meo['Date'] = pd.to_datetime(meo['Date']).dt.date
    meo['Date'] = meo['Date'].astype(str)

    meo['ViewData.Side0_UniqueIds'] = meo['ViewData.Side0_UniqueIds'].astype(str)
    meo['ViewData.Side1_UniqueIds'] = meo['ViewData.Side1_UniqueIds'].astype(str)

    meo['flag_side0'] = meo.apply(lambda x: len(x['ViewData.Side0_UniqueIds'].split(',')), axis=1)
    meo['flag_side1'] = meo.apply(lambda x: len(x['ViewData.Side1_UniqueIds'].split(',')), axis=1)

    meo.loc[meo['ViewData.Side0_UniqueIds']=='nan','flag_side0'] = 0
    meo.loc[meo['ViewData.Side1_UniqueIds']=='nan','flag_side1'] = 0

    meo.loc[meo['ViewData.Side0_UniqueIds']=='None','flag_side0'] = 0
    meo.loc[meo['ViewData.Side1_UniqueIds']=='None','flag_side1'] = 0
   
    meo['ViewData.BreakID'] = meo['ViewData.BreakID'].astype(int)
    meo = meo[meo['ViewData.BreakID']!=-1]           .reset_index()           .drop('index',1)
          
    meo['Side_0_1_UniqueIds'] = meo['ViewData.Side0_UniqueIds'].astype(str) +                                 meo['ViewData.Side1_UniqueIds'].astype(str)
                                
    meo = meo.sort_values(by=['ViewData.Transaction ID','ViewData.Transaction Type'],ascending = False)
    return(meo)
    
def cleaned_aua(fun_filepath_aua):
    aua = pd.read_csv(fun_filepath_aua)       .drop_duplicates()       .reset_index()       .drop('index',1)       .sort_values(by=['ViewData.Transaction ID','ViewData.Transaction Type'],ascending = False)

    aua = normalize_bp_acct_col_names(fun_df = aua)

    
    aua['Side_0_1_UniqueIds'] = aua['ViewData.Side0_UniqueIds'].astype(str) +                                 aua['ViewData.Side1_UniqueIds'].astype(str)
    
    return(aua)

def Acct_MEO_combination_file(fun_side, fun_cleaned_meo_df):
    if(fun_side == 'PB' or fun_side == 'BP' or fun_side == 'B-P' or fun_side == 'Prime Broker'):
        side_meo = fun_cleaned_meo_df[(fun_cleaned_meo_df['flag_side1'] >= 1) & (fun_cleaned_meo_df['flag_side0'] == 0)]
#        Currency_col_name = 'ViewData.B-P Currency'
    elif(fun_side == 'Acct' or fun_side == 'Accounting'):
        side_meo = fun_cleaned_meo_df[(fun_cleaned_meo_df['flag_side1'] == 0) & (fun_cleaned_meo_df['flag_side0'] >= 1)]
#        Currency_col_name = 'ViewData.Accounting Currency'
    else:
        print('The only options for side are on of the following : ')
        print('For Prime Broker side, the options are PB or BP or B-P or Prime Broker')
        print('For Accounting side, the options are Acct or Accounting')
        raise ValueError('Exiting function because fun_side argument was not from the accepted set of parameter values')
    
    side_meo['filter_key'] = side_meo['ViewData.Source Combination'].astype(str) +                          side_meo['ViewData.Mapped Custodian Account'].astype(str) +                          side_meo['ViewData.Currency'].astype(str)
        
    side_meo_training_df =[]
    for key in (list(np.unique(np.array(list(side_meo['filter_key'].values))))):
        side_meo_filter_slice = side_meo[side_meo['filter_key']==key]
        if side_meo_filter_slice.empty == False:
    
            side_meo_filter_slice = side_meo_filter_slice.reset_index()
            side_meo_filter_slice = side_meo_filter_slice.drop('index', 1)
    
            side_meo_filter_joined = pd.merge(side_meo_filter_slice, side_meo_filter_slice, on='filter_key')
            side_meo_training_df.append(side_meo_filter_joined)
    return(pd.concat(side_meo_training_df))
    
def identifying_closed_breaks_from_Trans_type(fun_side, fun_transaction_type_list, fun_side_meo_combination_df, fun_setup_code_crucial):
    if(fun_side == 'PB' or fun_side == 'BP' or fun_side == 'B-P' or fun_side == 'Prime Broker'):
        Net_amount_col_name_list = ['ViewData.B-P Net Amount_' + x for x in ['x','y']]
        Side_0_1_UniqueIds_col_name_list = ['ViewData.Side1_UniqueIds_' + x for x in ['x','y']]
    elif(fun_side == 'Acct' or fun_side == 'Accounting'):
        Net_amount_col_name_list = ['ViewData.Accounting Net Amount_' + x for x in ['x','y']]
        Side_0_1_UniqueIds_col_name_list = ['ViewData.Side0_UniqueIds_' + x for x in ['x','y']]
    else:
        print('The only options for side are on of the following : ')
        print('For Prime Broker side, the options are PB or BP or B-P or Prime Broker')
        print('For Accounting side, the options are Acct or Accounting')
        raise ValueError('Exiting function because fun_side argument was not from the accepted set of parameter values')        
    
    if(fun_setup_code_crucial == '379'):
        Transaction_type_closed_break_df =             fun_side_meo_combination_df[                     (fun_side_meo_combination_df['ViewData.Transaction Type_x'].astype(str).isin(fun_transaction_type_list)) &                     (fun_side_meo_combination_df['ViewData.Transaction Type_y'].astype(str).isin(fun_transaction_type_list)) &                     (abs(fun_side_meo_combination_df[Net_amount_col_name_list[0]]).astype(str) == abs(fun_side_meo_combination_df[Net_amount_col_name_list[1]]).astype(str)) &                     (fun_side_meo_combination_df[Side_0_1_UniqueIds_col_name_list[0]].astype(str) != fun_side_meo_combination_df[Side_0_1_UniqueIds_col_name_list[1]].astype(str))                     ]
    return(set(
                Transaction_type_closed_break_df['ViewData.Side0_UniqueIds_x'].astype(str) + \
                Transaction_type_closed_break_df['ViewData.Side1_UniqueIds_x'].astype(str)
               ))

def closed_breaks_captured_mode(fun_aua_df, fun_transaction_type, fun_captured_closed_breaks_set, fun_mode):
    if(fun_transaction_type != 'All_Closed_Breaks'):
        aua_df = fun_aua_df[(fun_aua_df['ViewData.Status'] == 'UCB') &                             (fun_aua_df['ViewData.Transaction Type'] == fun_transaction_type)]
    else:
        aua_df = fun_aua_df[(fun_aua_df['ViewData.Status'] == 'UCB')]
        
    aua_side_0_1_UniqueIds_set = set(aua_df['ViewData.Side0_UniqueIds'].astype(str) +                                  aua_df['ViewData.Side1_UniqueIds'].astype(str))
    if(fun_mode == 'Correctly_Captured_In_AUA'):
        list_to_return = list(aua_side_0_1_UniqueIds_set & fun_captured_closed_breaks_set)
    elif(fun_mode == 'Not_Captured_In_AUA'):
        list_to_return = list(aua_side_0_1_UniqueIds_set - fun_captured_closed_breaks_set)
    elif(fun_mode == 'Over_Captured_In_AUA'):
        list_to_return = list(fun_captured_closed_breaks_set - aua_side_0_1_UniqueIds_set)
    return(list_to_return)

def update_dict_to_output_breakids_number_pct(fun_dict, fun_aua_df, fun_loop_transaction_type, fun_count, fun_Side_0_1_UniqueIds_list):
    mode_type_list = ['Correctly_Captured_In_AUA','Not_Captured_In_AUA','Over_Captured_In_AUA']
    for mode_type in mode_type_list:
#    if(fun_loop_transaction_type != 'All_Closed_Breaks'):
        fun_dict[fun_loop_transaction_type][mode_type + '_BreakIDs_in_AUA'] = list(set(            fun_aua_df[fun_aua_df['Side_0_1_UniqueIds'].isin(                     closed_breaks_captured_mode(fun_aua_df = fun_aua_df,                                         fun_transaction_type = fun_loop_transaction_type,                                         fun_captured_closed_breaks_set = set(fun_Side_0_1_UniqueIds_list),                                         fun_mode = mode_type))]                    ['ViewData.BreakID']))
    
        fun_total_number = len(                             fun_dict[fun_loop_transaction_type][mode_type + '_BreakIDs_in_AUA'])
        
        fun_dict[fun_loop_transaction_type][mode_type + '_Total_Number'] = len(                             fun_dict[fun_loop_transaction_type][mode_type + '_BreakIDs_in_AUA'])
        
        if(fun_count != 0):
            
            fun_dict[fun_loop_transaction_type][mode_type + '_Percentage'] = fun_total_number/fun_count#\
#                                 fun_dict[fun_loop_transaction_type][mode_type + '_Total_Number']/fun_count
        
        else:
            fun_dict[fun_loop_transaction_type][mode_type + '_Percentage'] = fun_loop_transaction_type + ' not found in Closed breaks of AUA'
    return(fun_dict)

def closed_daily_run(fun_setup_code, 
                     fun_date, 
                     fun_meo_df_daily_run#,
#                     fun_main_filepath_meo, 
#                     fun_main_filepath_aua
                     ):
    setup_val = fun_setup_code
    main_meo = cleaned_meo(fun_meo_df = fun_meo_df_daily_run)#, fun_filepath_meo = fun_main_filepath_meo
    
    BP_meo_training_df = Acct_MEO_combination_file(fun_side = 'PB', \
                                                   fun_cleaned_meo_df = main_meo)
    
    Acct_meo_training_df = Acct_MEO_combination_file(fun_side = 'Acct', \
                                                     fun_cleaned_meo_df = main_meo)

#    main_aua = cleaned_aua(fun_filepath_aua = fun_main_filepath_aua)
    
    if(fun_setup_code == '379'):
        Transaction_Type_dict = {
                                'Interest BP_side' : {'side' : 'PB',
                                           'Transaction_Type' : ['Interest'],
                                           'Side_meo_training_df' : BP_meo_training_df},
                                'Interest Acct_side' : {'side' : 'Acct',
                                           'Transaction_Type' : ['Interest'],
                                           'Side_meo_training_df' : Acct_meo_training_df},
                                'STIF Interest BP_side' : {'side' : 'PB',
                                           'Transaction_Type' : ['STIF Interest'],
                                           'Side_meo_training_df' : BP_meo_training_df},
                                'STIF Interest Acct_side' : {'side' : 'Acct',
                                           'Transaction_Type' : ['STIF Interest'],
                                           'Side_meo_training_df' : Acct_meo_training_df},
                                'Buy BP_side' : {'side' : 'PB',
                                           'Transaction_Type' : ['Buy'],
                                           'Side_meo_training_df' : BP_meo_training_df},
                                'Buy Acct_side' : {'side' : 'Acct',
                                           'Transaction_Type' : ['Buy'],
                                           'Side_meo_training_df' : Acct_meo_training_df},
                                'Sell BP_side' : {'side' : 'PB',
                                           'Transaction_Type' : ['Sell'],
                                           'Side_meo_training_df' : BP_meo_training_df},
                                'Sell Acct_side' : {'side' : 'Acct',
                                           'Transaction_Type' : ['Sell'],
                                           'Side_meo_training_df' : Acct_meo_training_df},
                                'ForwardFX BP_side' : {'side' : 'PB',
                                           'Transaction_Type' : ['ForwardFX'],
                                           'Side_meo_training_df' : BP_meo_training_df},
                                'ForwardFX Acct_side' : {'side' : 'Acct',
                                           'Transaction_Type' : ['ForwardFX'],
                                           'Side_meo_training_df' : Acct_meo_training_df},
                                'Internal Trans' : {'side' : 'PB',
                                           'Transaction_Type' : ['Internal Trans'],
                                           'Side_meo_training_df' : BP_meo_training_df},
                                'Withdraw' : {'side' : 'Acct',
                                           'Transaction_Type' : ['Withdraw'],
                                           'Side_meo_training_df' : Acct_meo_training_df},
                                'Deposit' : {'side' : 'Acct',
                                           'Transaction_Type' : ['Deposit'],
                                           'Side_meo_training_df' : Acct_meo_training_df},
                                'Redemption' : {'side' : 'PB',
                                           'Transaction_Type' : ['Redemption'],
                                           'Side_meo_training_df' : BP_meo_training_df},
                                'Subscription' : {'side' : 'PB',
                                           'Transaction_Type' : ['Redemption'],
                                           'Side_meo_training_df' : BP_meo_training_df},
                                'Incoming Wire' : {'side' : 'PB',
                                           'Transaction_Type' : ['Incoming Wire'],
                                           'Side_meo_training_df' : BP_meo_training_df},
                                'Transfer' : {'side' : 'Acct',
                                           'Transaction_Type' : ['Transfer'],
                                           'Side_meo_training_df' : Acct_meo_training_df},
                                'Withdrawal BP_side' : {'side' : 'PB',
                                           'Transaction_Type' : ['Withdrawal'],
                                           'Side_meo_training_df' : BP_meo_training_df},
                                'Withdrawal Acct_side' : {'side' : 'Acct',
                                           'Transaction_Type' : ['Withdrawal'],
                                           'Side_meo_training_df' : Acct_meo_training_df},
                                'Revenue' : {'side' : 'Acct',
                                           'Transaction_Type' : ['Revenue'],
                                           'Side_meo_training_df' : Acct_meo_training_df},
                                'Pay Down' : {'side' : 'Acct',
                                           'Transaction_Type' : ['Pay Down'],
                                           'Side_meo_training_df' : Acct_meo_training_df},
                                'Over & Short' : {'side' : 'PB',
                                           'Transaction_Type' : ['Over & Short'],
                                           'Side_meo_training_df' : BP_meo_training_df}
                                }

    print(os.getcwd())
    os.chdir('D:\\ViteosModel\\Closed')
    print(os.getcwd())
    
    filepath_stdout = fun_setup_code + '_closed_run_date_' + str(fun_date) + '_timestamp_' + str(datetime.now().strftime("%d_%m_%Y_%H_%M")) + '.txt'
    orig_stdout = sys.stdout
    f = open(filepath_stdout, 'w')
    sys.stdout = f
    
    Side_0_1_UniqueIds_closed_all_list = []
    for Transaction_type in Transaction_Type_dict:

        Side_0_1_UniqueIds_for_Transaction_type = identifying_closed_breaks_from_Trans_type(fun_side = Transaction_Type_dict.get(Transaction_type).get('side'), \
                                                                                            fun_transaction_type_list = Transaction_Type_dict.get(Transaction_type).get('Transaction_Type'), \
                                                                                            fun_side_meo_combination_df = Transaction_Type_dict.get(Transaction_type).get('Side_meo_training_df'), \
                                                                                            fun_setup_code_crucial = setup_val)

#        count_closed_breaks_for_transaction_type = len(set(main_aua[(main_aua['ViewData.Status'] == 'UCB') & \
#                                                                    (main_aua['ViewData.Transaction Type'] == Transaction_type)]['Side_0_1_UniqueIds']))
#        
#        Transaction_Type_dict = update_dict_to_output_breakids_number_pct(fun_dict = Transaction_Type_dict, \
#                                                                          fun_aua_df = main_aua, \
#                                                                          fun_loop_transaction_type = Transaction_type, \
#                                                                          fun_count = count_closed_breaks_for_transaction_type, \
#                                                                          fun_Side_0_1_UniqueIds_list = Side_0_1_UniqueIds_for_Transaction_type)
            
        
        Side_0_1_UniqueIds_closed_all_list.extend(Side_0_1_UniqueIds_for_Transaction_type)
        print('\n' + Transaction_type + '\n')
#        pprint.pprint(dictionary_exclude_keys(fun_dict = Transaction_Type_dict.get(Transaction_type),                                      fun_keys_to_exclude = {'side','Transaction_Type','Side_meo_training_df'}),                      width = 4)
    
    sys.stdout = orig_stdout
    f.close()
    
#    count_all_closed_breaks = len(set(main_aua[(main_aua['ViewData.Status'] == 'UCB')]                                               ['Side_0_1_UniqueIds']))
    
#    aua_closed_dict = {'All_Closed_Breaks' : {}}
#    aua_closed_dict = update_dict_to_output_breakids_number_pct(fun_dict = aua_closed_dict,\
#                                                                fun_aua_df = main_aua, \
#                                                                fun_loop_transaction_type = 'All_Closed_Breaks', \
#                                                                fun_count = count_all_closed_breaks, \
#                                                                fun_Side_0_1_UniqueIds_list = Side_0_1_UniqueIds_closed_all_list)
    
#    write_dict_at_top(fun_filename = filepath_stdout, \
#                      fun_dict_to_add = aua_closed_dict)
    
    return(Side_0_1_UniqueIds_closed_all_list)

#### Closed break functions - End #### 

#### Break Prediction functions - Begin #### 

def equals_fun(a,b):
    if a == b:
        return 1
    else:
        return 0

vec_equals_fun = np.vectorize(equals_fun)


def descclean(com,cat_list):
    cat_all1 = []
    list1 = cat_list
    m = 0
    if (type(com) == str):
        com = com.lower()
        com1 =  re.split("[,/. \-!?:]+", com)
        
        
        
        for item in list1:
            if (type(item) == str):
                item = item.lower()
                item1 = item.split(' ')
                lst3 = [value for value in item1 if value in com1] 
                if len(lst3) == len(item1):
                    cat_all1.append(item)
                    m = m+1
            
                else:
                    m = m
            else:
                    m = 0
    else:
        m = 0
    

            
    if m >0 :
        return list(set(cat_all1))
    else:
        if ((type(com)==str)):
            if (len(com1)<4):
                if ((len(com1)==1) & com1[0].startswith('20')== True):
                    return 'swap id'
                else:
                    return com
            else:
                return 'NA'
        else:
            return 'NA'

def currcln(x):
    if (type(x)==list):
        return x
      
    else:
       
        
        if x == 'NA':
            return "NA"
        elif (('dollar' in x) | ('dollars' in x )):
            return 'dollar'
        elif (('pound' in x) | ('pounds' in x)):
            return 'pound'
        elif ('yen' in x):
            return 'yen'
        elif ('euro' in x) :
            return 'euro'
        else:
            return x

def catcln1(cat,df):
    ret = []
    if (type(cat)==list):
        
        if 'equity swap settlement' in cat:
            ret.append('equity swap settlement')
        #return 'equity swap settlement'
        elif 'equity swap' in cat:
            ret.append('equity swap settlement')
        #return 'equity swap settlement'
        elif 'swap settlement' in cat:
            ret.append('equity swap settlement')
        #return 'equity swap settlement'
        elif 'swap unwind' in cat:
            ret.append('swap unwind')
        #return 'swap unwind'
   
    
    
    
        else:
        
       
            for item in cat:
            
                a = df[df['Pairing']==item]['replace'].values[0]
                if a not in ret:
                    ret.append(a)
        return list(set(ret))
      
    else:
        return cat

def desccat(x):
    if isinstance(x, list):
        
        if 'equity swap settlement' in x:
            return 'swap settlement'
        elif 'collateral transfer' in x:
            return 'collateral transfer'
        elif 'dividend' in x:
            return 'dividend'
        elif (('loan' in x) & ('option' in x)):
            return 'option loan'
        
        elif (('interest' in x) & ('corp' in x) ):
            return 'corp loan'
        elif (('interest' in x) & ('loan' in x) ):
            return 'interest'
        else:
            return x[0]
    else:
        return x

def new_pf_mapping(x):
    if x=='GSIL':
        return 'GS'
    elif x == 'CITIGM':
        return 'CITI'
    elif x == 'JPMNA':
        return 'JPM'
    else:
        return x

def mhreplaced(item):
    word1 = []
    word2 = []
    if (type(item) == str):
    
        for items in item.split(' '):
            if (type(items) == str):
                items = items.lower()
                if items.isdigit() == False:
                    word1.append(items)
        
            
                for c in word1:
                    if c.endswith('MH')==False:
                        word2.append(c)
    
                words = ' '.join(word2)
                return words
    else:
        return item
    

def fundmatch(item):
    items = item.lower()
    items = item.replace(' ','') 
    return items

def is_num(item):
    try:
        float(item)
        return True
    except ValueError:
        return False

def is_date_format(item):
    try:
        parse(item, fuzzy=False)
        return True
    
    except ValueError:
        return False
    
def date_edge_cases(item):
    if len(item) == 5 and item[2] =='/' and is_num(item[:2]) and is_num(item[3:]):
        return True
    return False

def nan_fun(x):
    if x=='nan':
        return 1
    else:
        return 0

def a_keymatch(a_cusip, a_isin):
    
    pb_nan = 0
    a_common_key = 'NA' 
    if a_cusip=='nan' and a_isin =='nan':
        pb_nan =1
    elif(a_cusip!='nan' and a_isin == 'nan'):
        a_common_key = a_cusip
    elif(a_cusip =='nan' and a_isin !='nan'):
        a_common_key = a_isin
    else:
        a_common_key = a_isin
        
    return (pb_nan, a_common_key)

def b_keymatch(b_cusip, b_isin):
    accounting_nan = 0
    b_common_key = 'NA'
    if b_cusip =='nan' and b_isin =='nan':
        accounting_nan =1
    elif (b_cusip!='nan' and b_isin == 'nan'):
        b_common_key = b_cusip
    elif(b_cusip =='nan' and b_isin !='nan'):
        b_common_key = b_isin
    else:
        b_common_key = b_isin
    return (accounting_nan, b_common_key)


def nan_equals_fun(a,b):
    if a==1 and b==1:
        return 1
    else:
        return 0


def new_key_match_fun(a,b,c):
    if a==b and c==0:
        return 1
    else:
        return 0

def umr_seg(X_test):
    b_count = X_test.groupby(['SideB.ViewData.Side0_UniqueIds'])['Predicted_action'].value_counts().reset_index(name='count')
    b_unique = X_test.groupby(['SideB.ViewData.Side0_UniqueIds'])['Predicted_action'].unique().reset_index()
    
    b_unique['len'] = b_unique['Predicted_action'].str.len()
    b_count2 = pd.merge(b_count, b_unique.drop('Predicted_action',1), on='SideB.ViewData.Side0_UniqueIds', how='left')
    umr_table = b_count2[(b_count2['Predicted_action']=='UMR_One_to_One') & (b_count2['count']==1) & (b_count2['len']<=2)]
    return umr_table['SideB.ViewData.Side0_UniqueIds'].values

def no_pair_seg(X_test):
    
    b_side_agg = X_test.groupby(['SideB.ViewData.Side0_UniqueIds'])['Predicted_action_2'].unique().reset_index()
    a_side_agg = X_test.groupby(['SideA.ViewData.Side1_UniqueIds'])['Predicted_action_2'].unique().reset_index()
    
    b_side_agg['len'] = b_side_agg['Predicted_action_2'].str.len()
    b_side_agg['No_Pair_flag'] = b_side_agg['Predicted_action_2'].apply(lambda x: 1 if 'No-Pair' in x else 0)

    a_side_agg['len'] = a_side_agg['Predicted_action_2'].str.len()
    a_side_agg['No_Pair_flag'] = a_side_agg['Predicted_action_2'].apply(lambda x: 1 if 'No-Pair' in x else 0)
    
    no_pair_ids_b_side = b_side_agg[(b_side_agg['len']==1) & (b_side_agg['No_Pair_flag']==1)]['SideB.ViewData.Side0_UniqueIds'].values

    no_pair_ids_a_side = a_side_agg[(a_side_agg['len']==1) & (a_side_agg['No_Pair_flag']==1)]['SideA.ViewData.Side1_UniqueIds'].values
    
    return no_pair_ids_b_side, no_pair_ids_a_side
    
def one_to_one_umb(data):
    
    count = data['SideB.ViewData.Side0_UniqueIds'].value_counts().reset_index(name='count0')
    id0s = count[count['count0']==1]['index'].unique()
    id1s = data[data['SideB.ViewData.Side0_UniqueIds'].isin(id0s)]['SideA.ViewData.Side1_UniqueIds']
    
    count1 = data['SideA.ViewData.Side1_UniqueIds'].value_counts().reset_index(name='count1')
    final_ids = count1[(count1['count1']==1) & (count1['index'].isin(id1s))]['index'].unique()
    return final_ids

def sample_otm(data, col1, col2):
    ss =[]
    count_table = data[col2].value_counts().reset_index()
    count_table.columns = [col2, 'count']
    for i in data[col1].unique():
        ids1 = data[data[col1]==i][col2].unique()
        if count_table[count_table[col2].isin(ids1)]['count'].max() ==1:
            ss.append(i)
        else:
            pass
    return ss


def sample_mto(data, col1, col2):
    ss =[]
    count_table = data[col1].value_counts().reset_index()
    count_table.columns = [col1, 'count']
    for i in data[col2].unique():
        ids0 = data[data[col2]==i][col1].unique()
        if count_table[count_table[col1].isin(ids0)]['count'].max() ==1:
            ss.append(i)
        else:
            pass
    return ss

def find_closese_sum(numbers, targets):
    numbers = numbers[:]
    for t in targets:
        if not numbers:
            break
    combs = sum([list(itertools.combinations(numbers, r)) for r in range(1, len(numbers)+1)], [])
    sums = np.asarray(list(map(sum, combs)))
    bestcomb = combs[np.argmin(np.abs(np.asarray(sums) - t))]
    numbers = list(set(numbers).difference(bestcomb))
    print("Target: {},  combination: {}".format(t, bestcomb))

def finding_closet(ls,target,depth):
    closest = []
    for i in itertools.combinations(ls, depth):

        if sum(i) == target:
            return i
        else:
            closest.append((abs(sum(i) - target), i))
    return min(closest)[1]

def normalize_final_no_pair_table_col_names(fun_final_no_pair_table):
    final_no_pair_table_col_names_mapping_dict = {
                                      'SideA.ViewData.Side1_UniqueIds' : 'ViewData.Side1_UniqueIds',
                                      'SideB.ViewData.Side0_UniqueIds' : 'ViewData.Side0_UniqueIds',
                                      'SideA.ViewData.BreakID_A_side' : 'ViewData.BreakID_Side1', 
                                      'SideB.ViewData.BreakID_B_side' : 'ViewData.BreakID_Side0'
                                      }
    fun_final_no_pair_table.rename(columns = final_no_pair_table_col_names_mapping_dict, inplace = True)
    return(fun_final_no_pair_table)
    

#### Break Prediction functions - End #### 


date_numbers_list = [16]
                     #2,3,4,
                    # 7,8,9,10,11,
                    # 14,15,16,17,18,
                    # 21,22,23,24,25,
                    # 28,29,30]

client = 'OakTree'

setup_code = '379'

today = date.today()
d1 = datetime.strptime(today.strftime("%Y-%m-%d"),"%Y-%m-%d")
desired_date = d1 - timedelta(days=4)
desired_date_str = desired_date.strftime("%Y-%m-%d")
date_input = desired_date_str

filepaths_AUA = '//vitblrdevcons01/Raman  Strategy ML 2.0/All_Data/' + client + '/JuneData/AUA/AUACollections.AUA_HST_RecData_' + setup_code + '_' + str(date_input) + '.csv'
filepaths_MEO = '//vitblrdevcons01/Raman  Strategy ML 2.0/All_Data/' + client + '/JuneData/MEO/MeoCollections.MEO_HST_RecData_' + setup_code + '_' + str(date_input) + '.csv'
filepaths_no_pair_id_data = '//vitblrdevcons01/Raman  Strategy ML 2.0/All_Data/' + client + '/UAT_Run/X_Test_' + setup_code + '/no_pair_ids_' + setup_code + '_' + str(date_input) + '.csv'
filepaths_no_pair_id_no_data_warning = '//vitblrdevcons01/Raman  Strategy ML 2.0/All_Data/' + client + '/UAT_Run/X_Test_' + setup_code + '/WARNING_no_pair_ids_' + setup_code + str(date_input) + '.csv'


mngdb_obj_1_for_reading_and_writing_in_uat_server = mngdb(param_without_ssh  = True, param_without_RabbitMQ_pipeline = True,
                 param_SSH_HOST = None, param_SSH_PORT = None,
                 param_SSH_USERNAME = None, param_SSH_PASSWORD = None,
                 param_MONGO_HOST = '192.168.170.158', param_MONGO_PORT = 27017,
                 param_MONGO_USERNAME = '', param_MONGO_PASSWORD = '')
mngdb_obj_1_for_reading_and_writing_in_uat_server.connect_with_or_without_ssh()
#db_3_for_writing_in_ml_server = mngdb_obj_1_for_reading_and_writing_in_uat_server.client['MEO_AUA_Collections']
#db_4_for_MEO_data = mngdb_obj_3_for_writing_in_uat_server.client['MeoCollections']
db_1_for_MEO_data = mngdb_obj_1_for_reading_and_writing_in_uat_server.client['ReconDB_ML_Test']
#db_5_for_AUA_data = mngdb_obj_3_for_writing_in_uat_server.client['AUACollections']
#db_6_for_prediction = mngdb_obj_3_for_writing_in_uat_server.client['MLPrediction_Cash']


query_1_for_MEO_data = db_1_for_MEO_data['RecData_' + setup_code].find({ 
                                                                     "LastPerformedAction": 31
                                                             },
                                                             {
                                                                     "DataSides" : 1,
                                                                     "BreakID" : 1,
                                                                     "LastPerformedAction" : 1,
                                                                     "TaskInstanceID" : 1,
                                                                     "SourceCombinationCode" : 1,
                                                                     "MetaData" : 1, 
                                                                     "ViewData" : 1
                                                             })
list_of_dicts_query_result_1 = list(query_1_for_MEO_data)

meo_df = json_normalize(list_of_dicts_query_result_1)
meo_df = meo_df.loc[:,meo_df.columns.str.startswith('ViewData')]
meo_df['ViewData.Task Business Date'] = meo_df['ViewData.Task Business Date'].apply(dt.datetime.isoformat) 

meo = meo_df[new_cols]

Side_0_1_UniqueIds_closed_all_dates_list = []

i = 0
for i in range(0,len(date_numbers_list)):

    Side_0_1_UniqueIds_closed_all_dates_list.append(
            closed_daily_run(fun_setup_code=setup_code,\
                             fun_date = i,\
                             fun_meo_df_daily_run = meo)
#                             fun_main_filepath_meo= filepaths_MEO[i],\
#                             fun_main_filepath_aua = filepaths_AUA[i])
            )

new_closed_keys = [i.replace('nan','') for i in Side_0_1_UniqueIds_closed_all_dates_list[0]]
new_closed_keys = [i.replace('None','') for i in new_closed_keys]


df1 = meo[~meo['ViewData.Status'].isin(['SMT','HST', 'OC', 'CT', 'Archive','SMR'])]
#df = df[df['MatchStatus'] != 21]
df1 = df1[~df1['ViewData.Status'].isnull()]
df1 = df1.reset_index()
df1 = df1.drop('index',1)

#df1['close_key'] = df1['ViewData.Side0_UniqueIds'].astype(str) + df1['ViewData.Side1_UniqueIds'].astype(str)


## Output for Closed breaks

#closed_df = df1[df1['close_key'].isin(list(Side_0_1_UniqueIds_closed_all_dates_list[0]))]
closed_df_side1 = df1[df1['ViewData.Side1_UniqueIds'].isin(new_closed_keys)]
closed_df_side0 = df1[df1['ViewData.Side0_UniqueIds'].isin(new_closed_keys)]
closed_df = closed_df_side1.append(closed_df_side0)

df2 = df1[~((df1['ViewData.Side1_UniqueIds'].isin(new_closed_keys)) | (df1['ViewData.Side0_UniqueIds'].isin(new_closed_keys)))]
df = df2.copy()
df = df.reset_index()
df = df.drop('index',1)
df['Date'] = pd.to_datetime(df['ViewData.Task Business Date'])

df = df[~df['Date'].isnull()]
df = df.reset_index()
df = df.drop('index',1)

pd.to_datetime(df['Date'])

df['Date'] = pd.to_datetime(df['Date']).dt.date

df['Date'] = df['Date'].astype(str)

df = df[df['ViewData.Status'].isin(['OB','SDB','UOB','UDB','CMF','CNF','SMB'])]
df = df.reset_index()
df = df.drop('index',1)
df['ViewData.Side0_UniqueIds'] = df['ViewData.Side0_UniqueIds'].astype(str)
df['ViewData.Side1_UniqueIds'] = df['ViewData.Side1_UniqueIds'].astype(str)
df['flag_side0'] = df.apply(lambda x: len(x['ViewData.Side0_UniqueIds'].split(',')), axis=1)
df['flag_side1'] = df.apply(lambda x: len(x['ViewData.Side1_UniqueIds'].split(',')), axis=1)
df = df.rename(columns= {'ViewData.Cust Net Amount':'ViewData.B-P Net Amount'})

print('The Date value count is:')
print(df['Date'].value_counts())

date_i = df['Date'].mode()[0]

print('Choosing the date : ' + date_i)

sample = df[df['Date'] == date_i]
sample = sample.reset_index()
sample = sample.drop('index',1)

smb = sample[sample['ViewData.Status']=='SMB'].reset_index()
smb = smb.drop('index',1)
smb_pb = smb.copy()
smb_acc = smb.copy()
smb_pb['ViewData.Accounting Net Amount'] = np.nan
smb_pb['ViewData.Side0_UniqueIds'] = np.nan
smb_pb['ViewData.Status'] ='SMB-OB'

smb_acc['ViewData.B-P Net Amount'] = np.nan
smb_acc['ViewData.Side1_UniqueIds'] = np.nan
smb_acc['ViewData.Status'] ='SMB-OB'
sample = sample[sample['ViewData.Status']!='SMB']
sample = sample.reset_index()
sample = sample.drop('index',1)
sample = pd.concat([sample,smb_pb,smb_acc],axis=0)
sample = sample.reset_index()
sample = sample.drop('index',1)
sample['ViewData.Side0_UniqueIds'] = sample['ViewData.Side0_UniqueIds'].astype(str)
sample['ViewData.Side1_UniqueIds'] = sample['ViewData.Side1_UniqueIds'].astype(str)

sample.loc[sample['ViewData.Side0_UniqueIds']=='None','flag_side0'] = 0
sample.loc[sample['ViewData.Side1_UniqueIds']=='None','flag_side1'] = 0

sample.loc[sample['ViewData.Side1_UniqueIds']=='None','Trans_side'] = 'B_side'
sample.loc[sample['ViewData.Side0_UniqueIds']=='None','Trans_side'] = 'A_side'


sample.loc[sample['Trans_side']=='A_side','ViewData.B-P Currency'] = sample.loc[sample['Trans_side']=='A_side','ViewData.Currency']
sample.loc[sample['Trans_side']=='B_side','ViewData.Accounting Currency'] = sample.loc[sample['Trans_side']=='B_side','ViewData.Currency'] 

sample['ViewData.B-P Currency'] = sample['ViewData.B-P Currency'].astype(str)
sample['ViewData.Accounting Currency'] = sample['ViewData.Accounting Currency'].astype(str)
sample['ViewData.Mapped Custodian Account'] = sample['ViewData.Mapped Custodian Account'].astype(str)
#sample['ViewData.Mapped Custodian Account'] = sample['ViewData.Mapped Custodian Account'].astype(str)
sample['filter_key'] = sample.apply(lambda x: x['ViewData.Mapped Custodian Account'] + x['ViewData.B-P Currency'] if x['Trans_side']=='A_side' else x['ViewData.Mapped Custodian Account'] + x['ViewData.Accounting Currency'], axis=1)
sample1 = sample[(sample['flag_side0']<=1) & (sample['flag_side1']<=1) & (sample['ViewData.Status'].isin(['OB','SPM','SDB','UDB','UOB','SMB-OB','CNF','CMF']))]
sample1 = sample1.reset_index()
sample1 = sample1.drop('index', 1)
sample1['ViewData.BreakID'] = sample1['ViewData.BreakID'].astype(int)
sample1 = sample1[sample1['ViewData.BreakID']!=-1]
sample1 = sample1.reset_index()
sample1 = sample1.drop('index',1)
sample1 = sample1.sort_values(['ViewData.BreakID','Date'], ascending =[True, False])
sample1 = sample1.reset_index()
sample1 = sample1.drop('index',1)
aa = sample1[sample1['Trans_side']=='A_side']
bb = sample1[sample1['Trans_side']=='B_side']
aa['filter_key'] = aa['ViewData.Source Combination'].astype(str) + aa['ViewData.Mapped Custodian Account'].astype(str) + aa['ViewData.B-P Currency'].astype(str)

bb['filter_key'] = bb['ViewData.Source Combination'].astype(str) + bb['ViewData.Mapped Custodian Account'].astype(str) + bb['ViewData.Accounting Currency'].astype(str)
aa = aa.reset_index()
aa = aa.drop('index', 1)
bb = bb.reset_index()
bb = bb.drop('index', 1)
bb = bb[~bb['ViewData.Accounting Net Amount'].isnull()]
bb = bb.reset_index()
bb = bb.drop('index',1)
###################### loop m*n ###############################



pool =[]
key_index =[]
training_df =[]

no_pair_ids = []
#max_rows = 5

for d in tqdm(aa['Date'].unique()):
    aa1 = aa.loc[aa['Date']==d,:][common_cols]
    bb1 = bb.loc[bb['Date']==d,:][common_cols]
    
    aa1 = aa1.reset_index()
    aa1 = aa1.drop('index',1)
    bb1 = bb1.reset_index()
    bb1 = bb1.drop('index', 1)
    
    bb1 = bb1.sort_values(by='filter_key',ascending =True)
    
    for key in (list(np.unique(np.array(list(aa1['filter_key'].values) + list(bb1['filter_key'].values))))):
        
        df1 = aa1[aa1['filter_key']==key]
        df2 = bb1[bb1['filter_key']==key]

        if df1.empty == False and df2.empty == False:
            #aa_df = pd.concat([aa1[aa1.index==i]]*repeat_num, ignore_index=True)
            #bb_df = bb1.loc[pool[len(pool)-1],:][common_cols].reset_index()
            #bb_df = bb_df.drop('index', 1)

            df1 = df1.rename(columns={'ViewData.BreakID':'ViewData.BreakID_A_side'})
            df2 = df2.rename(columns={'ViewData.BreakID':'ViewData.BreakID_B_side'})

            #dff  = pd.concat([aa[aa.index==i],bb.loc[pool[i],:][accounting_vars]],axis=1)

            df1 = df1.reset_index()
            df2 = df2.reset_index()
            df1 = df1.drop('index', 1)
            df2 = df2.drop('index', 1)

            df1.columns = ['SideA.' + x  for x in df1.columns] 
            df2.columns = ['SideB.' + x  for x in df2.columns]

            df1 = df1.rename(columns={'SideA.filter_key':'filter_key'})
            df2 = df2.rename(columns={'SideB.filter_key':'filter_key'})

            #dff = pd.concat([aa_df,bb_df],axis=1)
            dff = merge(df1, df2, on='filter_key')
            training_df.append(dff)
                #key_index.append(i)
            #else:
            #no_pair_ids.append([aa1[(aa1['filter_key']=='key') & (aa1['ViewData.Status'].isin(['OB','SDB']))]['ViewData.Side1_UniqueIds'].values[0]])
               # no_pair_ids.append(aa1[(aa1['filter_key']== key) & (aa1['ViewData.Status'].isin(['OB','SDB']))]['ViewData.Side1_UniqueIds'].values[0])
    
        else:
            no_pair_ids.append([aa1[(aa1['filter_key']==key) & (aa1['ViewData.Status'].isin(['OB','SDB']))]['ViewData.Side1_UniqueIds'].values])
            no_pair_ids.append([bb1[(bb1['filter_key']==key) & (bb1['ViewData.Status'].isin(['OB','SDB']))]['ViewData.Side0_UniqueIds'].values])
            


if len(no_pair_ids) != 0:
    no_pair_ids = np.unique(np.concatenate(no_pair_ids,axis=1)[0])
    no_pair_ids_df = pd.DataFrame(no_pair_ids, columns = ['Side0_1_UniqueIds'])
#    no_pair_ids_df = pd.merge(no_pair_ids_df, meo_df[['ViewData.Side1_UniqueIds','ViewData.BreakID','ViewData.Task ID','ViewData.Task Business Date']].drop_duplicates(), left_on = 'Side0_1_UniqueIds',right_on = 'ViewData.Side1_UniqueIds', how='left')
#    no_pair_ids_df = pd.merge(no_pair_ids_df, meo_df[['ViewData.Side0_UniqueIds','ViewData.BreakID','ViewData.Task ID','ViewData.Task Business Date']].drop_duplicates(), left_on = 'Side0_1_UniqueIds',right_on = 'ViewData.Side0_UniqueIds', how='left')
#    #no_pair_ids_df = no_pair_ids_df.rename(columns={'0':'filter_key'})
#    no_pair_ids_df['Predicted_Status'] = 'OB'
#    no_pair_ids_df['Predicted_action'] = 'No-Pair'
#    no_pair_ids_df['probability_No_pair'] = 0.9933
#    no_pair_ids_df['probability_UMB'] = 0.0033
#    no_pair_ids_df['probability_UMR'] = 0.0033    
#    no_pair_ids_df['ML_flag'] = 'ML'
#    no_pair_ids_df['TaskID'] = setup_code 
    no_pair_ids_df.to_csv(filepaths_no_pair_id_data)
else:
     with open(filepaths_no_pair_id_no_data_warning, 'w') as f:
         f.write('No no pair ids found for this setup and date combination')


test_file = pd.concat(training_df)

test_file = test_file.reset_index()
test_file = test_file.drop('index',1)

test_file['SideB.ViewData.BreakID_B_side'] = test_file['SideB.ViewData.BreakID_B_side'].astype('int64')
test_file['SideA.ViewData.BreakID_A_side'] = test_file['SideA.ViewData.BreakID_A_side'].astype('int64')
test_file['SideB.ViewData.CUSIP'] = test_file['SideB.ViewData.CUSIP'].str.split(".",expand=True)[0]
test_file['SideA.ViewData.CUSIP'] = test_file['SideA.ViewData.CUSIP'].str.split(".",expand=True)[0]
test_file['SideA.ViewData.ISIN'] = test_file['SideA.ViewData.ISIN'].astype(str)
test_file['SideB.ViewData.ISIN'] = test_file['SideB.ViewData.ISIN'].astype(str)
test_file['SideA.ViewData.CUSIP'] = test_file['SideA.ViewData.CUSIP'].astype(str)
test_file['SideB.ViewData.CUSIP'] = test_file['SideB.ViewData.CUSIP'].astype(str)
test_file['SideA.ViewData.Currency'] = test_file['SideA.ViewData.Currency'].astype(str)
test_file['SideB.ViewData.Currency'] = test_file['SideB.ViewData.Currency'].astype(str)


test_file['SideA.ViewData.Trade Date'] = test_file['SideA.ViewData.Trade Date'].astype(str)
test_file['SideB.ViewData.Trade Date'] = test_file['SideB.ViewData.Trade Date'].astype(str)
test_file['SideA.ViewData.Settle Date'] = test_file['SideA.ViewData.Settle Date'].astype(str)
test_file['SideB.ViewData.Settle Date'] = test_file['SideB.ViewData.Settle Date'].astype(str)
test_file['SideA.ViewData.Fund'] = test_file['SideA.ViewData.Fund'].astype(str)
test_file['SideB.ViewData.Fund'] = test_file['SideB.ViewData.Fund'].astype(str)

values_ISIN_A_Side = test_file['SideA.ViewData.ISIN'].values
values_ISIN_B_Side = test_file['SideB.ViewData.ISIN'].values
#test_file['ISIN_match'] = vec_equals_fun(values_ISIN_A_Side,values_ISIN_B_Side)

values_CUSIP_A_Side = test_file['SideA.ViewData.CUSIP'].values
values_CUSIP_B_Side = test_file['SideB.ViewData.CUSIP'].values
#
# values_CUSIP_A_Side = test_file['SideA.ViewData.Currency'].values
# values_CUSIP_B_Side = test_file['SideB.ViewData.Currency'].values

values_Currency_match_A_Side = test_file['SideA.ViewData.Currency'].values
values_Currency_match_B_Side = test_file['SideA.ViewData.Currency'].values

values_Trade_Date_match_A_Side = test_file['SideA.ViewData.Trade Date'].values
values_Trade_Date_match_B_Side = test_file['SideB.ViewData.Trade Date'].values

values_Settle_Date_match_A_Side = test_file['SideA.ViewData.Settle Date'].values
values_Settle_Date_match_B_Side = test_file['SideB.ViewData.Settle Date'].values

values_Fund_match_A_Side = test_file['SideA.ViewData.Fund'].values
values_Fund_match_B_Side = test_file['SideB.ViewData.Fund'].values

test_file['ISIN_match'] = vec_equals_fun(values_ISIN_A_Side,values_ISIN_B_Side)
test_file['CUSIP_match'] = vec_equals_fun(values_CUSIP_A_Side,values_CUSIP_B_Side)
test_file['Currency_match'] = vec_equals_fun(values_Currency_match_A_Side,values_Currency_match_B_Side)
test_file['Trade_Date_match'] = vec_equals_fun(values_Trade_Date_match_A_Side,values_Trade_Date_match_B_Side)
test_file['Settle_Date_match'] = vec_equals_fun(values_Settle_Date_match_A_Side,values_Settle_Date_match_B_Side)
test_file['Fund_match'] = vec_equals_fun(values_Fund_match_A_Side,values_Fund_match_B_Side)

test_file['Amount_diff_1'] = test_file['SideA.ViewData.Accounting Net Amount'] - test_file['SideB.ViewData.B-P Net Amount']
test_file['Amount_diff_2'] = test_file['SideB.ViewData.Accounting Net Amount'] - test_file['SideA.ViewData.B-P Net Amount']


# ## Description code

os.chdir('D:\\ViteosModel\\OakTree - Pratik Code')
print(os.getcwd())

## TODO - Import a csv file for description category mapping

com = pd.read_csv('desc cat with naveen oaktree.csv')
cat_list = list(set(com['Pairing']))



test_file['SideA.desc_cat'] = test_file['SideA.ViewData.Description'].apply(lambda x : descclean(x,cat_list))
test_file['SideB.desc_cat'] = test_file['SideB.ViewData.Description'].apply(lambda x : descclean(x,cat_list))

test_file['SideA.desc_cat'] = test_file['SideA.desc_cat'].apply(lambda x : currcln(x))
test_file['SideB.desc_cat'] = test_file['SideB.desc_cat'].apply(lambda x : currcln(x))

com = com.drop(['var','Catogery'], axis = 1)

com = com.drop_duplicates()

com['Pairing'] = com['Pairing'].apply(lambda x : x.lower())
com['replace'] = com['replace'].apply(lambda x : x.lower())


test_file['SideA.new_desc_cat'] = test_file['SideA.desc_cat'].apply(lambda x : catcln1(x,com))
test_file['SideB.new_desc_cat'] = test_file['SideB.desc_cat'].apply(lambda x : catcln1(x,com))

comp = ['inc','stk','corp ','llc','pvt','plc']
test_file['SideA.new_desc_cat'] = test_file['SideA.new_desc_cat'].apply(lambda x : 'Company' if x in comp else x)

test_file['SideB.new_desc_cat'] = test_file['SideB.new_desc_cat'].apply(lambda x : 'Company' if x in comp else x)

test_file['SideA.new_desc_cat'] = test_file['SideA.new_desc_cat'].apply(lambda x : desccat(x))
test_file['SideB.new_desc_cat'] = test_file['SideB.new_desc_cat'].apply(lambda x : desccat(x))
# ## Prime Broker
test_file['new_pb'] = test_file['SideA.ViewData.Mapped Custodian Account'].apply(lambda x : x.split('_')[0] if type(x)==str else x)
new_pb_mapping = {'GSIL':'GS','CITIGM':'CITI','JPMNA':'JPM'}
test_file['SideA.ViewData.Prime Broker'] = test_file['SideA.ViewData.Prime Broker'].fillna('kkk')
test_file['new_pb1'] = test_file.apply(lambda x : x['new_pb'] if x['SideA.ViewData.Prime Broker']=='kkk' else x['SideA.ViewData.Prime Broker'],axis = 1)
test_file['Trade_date_diff'] = (pd.to_datetime(test_file['SideA.ViewData.Trade Date']) - pd.to_datetime(test_file['SideB.ViewData.Trade Date'])).dt.days

test_file['Settle_date_diff'] = (pd.to_datetime(test_file['SideA.ViewData.Settle Date']) - pd.to_datetime(test_file['SideB.ViewData.Settle Date'])).dt.days

############ Fund match new ########

values_Fund_match_A_Side = test_file['SideA.ViewData.Fund'].values
values_Fund_match_B_Side = test_file['SideB.ViewData.Fund'].values

vec_fund_match = np.vectorize(fundmatch)

test_file['SideA.ViewData.Fund'] = vec_fund_match(values_Fund_match_A_Side)
test_file['SideB.ViewData.Fund'] = vec_fund_match(values_Fund_match_B_Side)

### New code for cleaning text variables 
trans_type_A_side = test_file['SideA.ViewData.Transaction Type']
trans_type_B_side = test_file['SideB.ViewData.Transaction Type']

asset_type_cat_A_side = test_file['SideA.ViewData.Asset Type Category']
asset_type_cat_B_side = test_file['SideB.ViewData.Asset Type Category']

invest_type_A_side = test_file['SideA.ViewData.Investment Type']
invest_type_B_side = test_file['SideB.ViewData.Investment Type']

prime_broker_A_side = test_file['SideA.ViewData.Prime Broker']
prime_broker_B_side = test_file['SideB.ViewData.Prime Broker']

# LOWER CASE
trans_type_A_side = [str(item).lower() for item in trans_type_A_side]
trans_type_B_side = [str(item).lower() for item in trans_type_B_side]

asset_type_cat_A_side = [str(item).lower() for item in asset_type_cat_A_side]
asset_type_cat_B_side = [str(item).lower() for item in asset_type_cat_B_side]

invest_type_A_side = [str(item).lower() for item in invest_type_A_side]
invest_type_B_side = [str(item).lower() for item in invest_type_B_side]

prime_broker_A_side = [str(item).lower() for item in prime_broker_A_side]
prime_broker_B_side = [str(item).lower() for item in prime_broker_B_side]

split_trans_A_side = [item.split() for item in trans_type_A_side]
split_trans_B_side = [item.split() for item in trans_type_B_side]

split_asset_A_side = [item.split() for item in asset_type_cat_A_side]
split_asset_B_side = [item.split() for item in asset_type_cat_B_side]

split_invest_A_side = [item.split() for item in invest_type_A_side]
split_invest_B_side = [item.split() for item in invest_type_B_side]

split_prime_A_side = [item.split() for item in prime_broker_A_side]
split_prime_b_side = [item.split() for item in prime_broker_B_side]

## Transacion type

remove_nums_A_side = [[item for item in sublist if not is_num(item)] for sublist in split_trans_A_side]
remove_nums_B_side = [[item for item in sublist if not is_num(item)] for sublist in split_trans_B_side]

remove_dates_A_side = [[item for item in sublist if not (is_date_format(item) or date_edge_cases(item))] for sublist in remove_nums_A_side]
remove_dates_B_side = [[item for item in sublist if not (is_date_format(item) or date_edge_cases(item))] for sublist in remove_nums_B_side]


# Specific to clients already used on, will have to be edited for other edge cases
remove_amts_A_side = [[item for item in sublist if item[0] != '$'] for sublist in remove_dates_A_side]
remove_amts_B_side = [[item for item in sublist if item[0] != '$'] for sublist in remove_dates_B_side]


clean_adr_A_side = [(['ADR'] if 'adr' in item else item) for item in remove_amts_A_side]
clean_adr_B_side = [(['ADR'] if 'adr' in item else item) for item in remove_amts_B_side]

clean_tax_A_side = [(item[:2] if '30%' in item else item) for item in clean_adr_A_side]
clean_tax_B_side = [(item[:2] if '30%' in item else item) for item in clean_adr_B_side]

remove_ons_A_side = [(item[:item.index('on')] if 'on' in item else item) for item in clean_tax_A_side]
remove_ons_B_side = [(item[:item.index('on')] if 'on' in item else item) for item in clean_tax_B_side]

clean_eqswap_A_side = [(item[1:] if 'eqswap' in item else item) for item in remove_ons_A_side]
clean_eqswap_B_side = [(item[1:] if 'eqswap' in item else item) for item in remove_ons_B_side]

remove_mh_A_side = [[item for item in sublist if 'mh' not in item] for sublist in clean_eqswap_A_side]
remove_mh_B_side = [[item for item in sublist if 'mh' not in item] for sublist in clean_eqswap_B_side]

remove_ats_A_side = [(item[:item.index('@')] if '@' in item else item) for item in remove_mh_A_side]
remove_ats_B_side = [(item[:item.index('@')] if '@' in item else item) for item in remove_mh_B_side]

cleaned_trans_types_A_side = [' '.join(item) for item in remove_ats_A_side]
cleaned_trans_types_B_side = [' '.join(item) for item in remove_ats_B_side]

# # INVESTMENT TYPE

remove_nums_i_A_side = [[item for item in sublist if not is_num(item)] for sublist in split_invest_A_side]
remove_nums_i_B_side = [[item for item in sublist if not is_num(item)] for sublist in split_invest_B_side]

remove_dates_i_A_side = [[item for item in sublist if not is_date_format(item)] for sublist in remove_nums_i_A_side]
remove_dates_i_B_side = [[item for item in sublist if not is_date_format(item)] for sublist in remove_nums_i_B_side]

cleaned_invest_A_side = [' '.join(item) for item in remove_dates_i_A_side]
cleaned_invest_B_side = [' '.join(item) for item in remove_dates_i_B_side]

remove_nums_a_A_side = [[item for item in sublist if not is_num(item)] for sublist in split_asset_A_side]
remove_nums_a_B_side = [[item for item in sublist if not is_num(item)] for sublist in split_asset_B_side]

remove_dates_a_A_side = [[item for item in sublist if not is_date_format(item)] for sublist in remove_nums_a_A_side]
remove_dates_a_B_side = [[item for item in sublist if not is_date_format(item)] for sublist in remove_nums_a_B_side]

cleaned_asset_A_side = [' '.join(item) for item in remove_dates_a_A_side]
cleaned_asset_B_side = [' '.join(item) for item in remove_dates_a_B_side]

test_file['SideA.ViewData.Transaction Type'] = cleaned_trans_types_A_side
test_file['SideB.ViewData.Transaction Type'] = cleaned_trans_types_B_side

test_file['SideA.ViewData.Investment Type'] = cleaned_invest_A_side
test_file['SideB.ViewData.Investment Type'] = cleaned_invest_B_side

test_file['SideA.ViewData.Asset Category Type'] = cleaned_asset_A_side
test_file['SideB.ViewData.Asset Category Type'] = cleaned_asset_B_side

values_transaction_type_match_A_Side = test_file['SideA.ViewData.Transaction Type'].values
values_transaction_type_match_B_Side = test_file['SideB.ViewData.Transaction Type'].values

vec_tt_match = np.vectorize(mhreplaced)

test_file['SideA.ViewData.Transaction Type'] = vec_tt_match(values_transaction_type_match_A_Side)
test_file['SideB.ViewData.Transaction Type'] = vec_tt_match(values_transaction_type_match_B_Side)

test_file.loc[test_file['SideA.ViewData.Transaction Type']=='int','SideA.ViewData.Transaction Type'] = 'interest'
test_file.loc[test_file['SideA.ViewData.Transaction Type']=='wires','SideA.ViewData.Transaction Type'] = 'wire'
test_file.loc[test_file['SideA.ViewData.Transaction Type']=='dividends','SideA.ViewData.Transaction Type'] = 'dividend'
test_file.loc[test_file['SideA.ViewData.Transaction Type']=='miscellaneous','SideA.ViewData.Transaction Type'] = 'misc'
test_file.loc[test_file['SideA.ViewData.Transaction Type']=='div','SideA.ViewData.Transaction Type'] = 'dividend'

test_file['SideA.ViewData.Investment Type'] = test_file['SideA.ViewData.Investment Type'].apply(lambda x: x.replace('eqty','equity'))
test_file['SideA.ViewData.Investment Type'] = test_file['SideA.ViewData.Investment Type'].apply(lambda x: x.replace('options','option'))
test_file['SideA.ViewData.Investment Type'] = test_file['SideA.ViewData.Investment Type'].apply(lambda x: x.replace('eqt','equity'))
test_file['SideA.ViewData.Investment Type'] = test_file['SideA.ViewData.Investment Type'].apply(lambda x: x.replace('eqty','equity'))

test_file['ViewData.Combined Transaction Type'] = test_file['SideA.ViewData.Transaction Type'].astype(str) +  test_file['SideB.ViewData.Transaction Type'].astype(str)
test_file['ViewData.Combined Fund'] = test_file['SideA.ViewData.Fund'].astype(str) + test_file['SideB.ViewData.Fund'].astype(str)

test_file['Combined_Investment_Type'] = test_file['SideA.ViewData.Investment Type'].astype(str) + test_file['SideB.ViewData.Investment Type'].astype(str)

test_file['Combined_Asset_Type_Category'] = test_file['SideA.ViewData.Asset Category Type'].astype(str) + test_file['SideB.ViewData.Asset Category Type'].astype(str)

    
vec_nan_fun = np.vectorize(nan_fun)
values_ISIN_A_Side = test_file['SideA.ViewData.ISIN'].values
values_ISIN_B_Side = test_file['SideB.ViewData.ISIN'].values
test_file['SideA.ISIN_NA'] = vec_nan_fun(values_ISIN_A_Side)
test_file['SideB.ISIN_NA'] = vec_nan_fun(values_ISIN_A_Side)

len(test_file['SideB.ViewData.CUSIP'].values)


    
vec_a_key_match_fun = np.vectorize(a_keymatch)
vec_b_key_match_fun = np.vectorize(b_keymatch)

values_ISIN_A_Side = test_file['SideA.ViewData.ISIN'].values
values_ISIN_B_Side = test_file['SideB.ViewData.ISIN'].values

values_CUSIP_A_Side = test_file['SideA.ViewData.CUSIP'].values
values_CUSIP_B_Side = test_file['SideB.ViewData.CUSIP'].values

test_file['SideB.ViewData.key_NAN']= vec_a_key_match_fun(values_CUSIP_B_Side,values_ISIN_B_Side)[0]
test_file['SideB.ViewData.Common_key'] = vec_a_key_match_fun(values_CUSIP_B_Side,values_ISIN_B_Side)[1]
test_file['SideA.ViewData.key_NAN'] = vec_b_key_match_fun(values_CUSIP_A_Side,values_ISIN_A_Side)[0]
test_file['SideA.ViewData.Common_key'] = vec_b_key_match_fun(values_CUSIP_A_Side,values_ISIN_A_Side)[1]

vec_nan_equal_fun = np.vectorize(nan_equals_fun)
values_key_NAN_B_Side = test_file['SideB.ViewData.key_NAN'].values
values_key_NAN_A_Side = test_file['SideA.ViewData.key_NAN'].values
test_file['All_key_nan'] = vec_nan_equal_fun(values_key_NAN_B_Side,values_key_NAN_A_Side )

test_file['SideB.ViewData.Common_key'] = test_file['SideB.ViewData.Common_key'].astype(str)
test_file['SideA.ViewData.Common_key'] = test_file['SideA.ViewData.Common_key'].astype(str)

vec_new_key_match_fun = np.vectorize(new_key_match_fun)
values_Common_key_B_Side = test_file['SideB.ViewData.Common_key'].values
values_Common_key_A_Side = test_file['SideA.ViewData.Common_key'].values
values_All_key_NAN = test_file['All_key_nan'].values

test_file['new_key_match']= vec_new_key_match_fun(values_Common_key_B_Side,values_Common_key_A_Side,values_All_key_NAN)

test_file['amount_percent'] = (test_file['SideA.ViewData.B-P Net Amount']/test_file['SideB.ViewData.Accounting Net Amount']*100)

test_file['SideB.ViewData.Investment Type'] = test_file['SideB.ViewData.Investment Type'].apply(lambda x: str(x).lower())
test_file['SideA.ViewData.Investment Type'] = test_file['SideA.ViewData.Investment Type'].apply(lambda x: str(x).lower())

test_file['SideB.ViewData.Prime Broker'] = test_file['SideB.ViewData.Prime Broker'].apply(lambda x: str(x).lower())
test_file['SideA.ViewData.Prime Broker'] = test_file['SideA.ViewData.Prime Broker'].apply(lambda x: str(x).lower())

test_file['SideB.ViewData.Asset Type Category'] = test_file['SideB.ViewData.Asset Type Category'].apply(lambda x: str(x).lower())
test_file['SideA.ViewData.Asset Type Category'] = test_file['SideA.ViewData.Asset Type Category'].apply(lambda x: str(x).lower())

test_file['ViewData.Combined Transaction Type'] = test_file['ViewData.Combined Transaction Type'].apply(lambda x: x.replace('jnl','journal'))

test_file['SideA.ViewData.Transaction Type'] = test_file['SideA.ViewData.Transaction Type'].apply(lambda x: x.replace('cover short','covershort'))

trade_types_A = ['buy', 'sell', 'covershort','sellshort',
       'fx', 'fx settlement', 'sell short',
       'trade not to be reported_buy', 'covershort','ptbl','ptss', 'ptcs', 'ptcl']
trade_types_B = ['trade not to be reported_buy','buy', 'sellshort', 'sell', 'covershort',
       'spotfx', 'forwardfx',
       'trade not to be reported_sell',
       'trade not to be reported_sellshort',
       'trade not to be reported_covershort']

test_file['SideA.TType'] = test_file.apply(lambda x: "Trade" if x['SideA.ViewData.Transaction Type'] in trade_types_A else "Non-Trade", axis=1)
test_file['SideB.TType'] = test_file.apply(lambda x: "Trade" if x['SideB.ViewData.Transaction Type'] in trade_types_B else "Non-Trade", axis=1)

test_file['Combined_Desc'] = test_file['SideA.new_desc_cat'] + test_file['SideB.new_desc_cat']

test_file['Combined_TType'] = test_file['SideA.TType'].astype(str) + test_file['SideB.TType'].astype(str)

for feature in ['SideA.Date','SideB.Date','SideA.ViewData.Settle Date','SideB.ViewData.Settle Date']:
    #train_full_new12[feature] = le.fit_transform(train_full_new12[feature])
    test_file[feature] = pd.to_datetime(test_file[feature],errors = 'coerce').dt.weekday



# ## UMR Mapping
## TODO Import HIstorical UMR FILE for Transaction Type mapping
oaktree_umr = pd.read_csv('OakTree_UMR.csv')

test_file['tt_map_flag'] = test_file.apply(lambda x: 1 if x['ViewData.Combined Transaction Type'] in oaktree_umr['ViewData.Combined Transaction Type'].unique() else 0, axis=1)

test_file['abs_amount_flag'] = test_file.apply(lambda x: 1 if x['SideB.ViewData.Accounting Net Amount'] == x['SideA.ViewData.B-P Net Amount']*(-1) else 0, axis=1)

test_file = test_file[~test_file['SideB.ViewData.Settle Date'].isnull()]
test_file = test_file[~test_file['SideA.ViewData.Settle Date'].isnull()]

test_file = test_file.reset_index().drop('index',1)
test_file['SideA.ViewData.Settle Date'] = test_file['SideA.ViewData.Settle Date'].astype(int)
test_file['SideB.ViewData.Settle Date'] = test_file['SideB.ViewData.Settle Date'].astype(int)


# ## Test file served into the model

test_file2 = test_file.copy()

X_test = test_file2[model_cols]

X_test = X_test.reset_index()
X_test = X_test.drop('index',1)
X_test = X_test.fillna(0)

X_test = X_test.fillna(0)

X_test.shape

X_test = X_test.drop_duplicates()
X_test = X_test.reset_index()
X_test = X_test.drop('index',1)

X_test.shape

# ## Model Pickle file import
## TODO Import Pickle file for 1st Model

filename = 'OakTree_final_model2.sav'

clf = pickle.load(open(filename, 'rb'))

# ## Predictions

# Actual class predictions
rf_predictions = clf.predict(X_test.drop(['SideB.ViewData.Status','SideB.ViewData.BreakID_B_side', 'SideA.ViewData.Status','SideA.ViewData.BreakID_A_side','SideA.ViewData._ID','SideB.ViewData._ID','SideB.ViewData.Side0_UniqueIds','SideA.ViewData.Side1_UniqueIds'],1))
# Probabilities for each class
rf_probs = clf.predict_proba(X_test.drop(['SideB.ViewData.Status','SideB.ViewData.BreakID_B_side', 'SideA.ViewData.Status','SideA.ViewData.BreakID_A_side','SideA.ViewData._ID','SideB.ViewData._ID','SideB.ViewData.Side0_UniqueIds','SideA.ViewData.Side1_UniqueIds'],1))[:, 1]

probability_class_0 = clf.predict_proba(X_test.drop(['SideB.ViewData.Status','SideB.ViewData.BreakID_B_side','SideA.ViewData.Status','SideA.ViewData.BreakID_A_side','SideA.ViewData._ID','SideB.ViewData._ID','SideB.ViewData.Side0_UniqueIds','SideA.ViewData.Side1_UniqueIds'],1))[:, 0]
probability_class_1 = clf.predict_proba(X_test.drop(['SideB.ViewData.Status','SideB.ViewData.BreakID_B_side', 'SideA.ViewData.Status','SideA.ViewData.BreakID_A_side','SideA.ViewData._ID','SideB.ViewData._ID','SideB.ViewData.Side0_UniqueIds','SideA.ViewData.Side1_UniqueIds'],1))[:, 1]

probability_class_2 = clf.predict_proba(X_test.drop(['SideB.ViewData.Status','SideB.ViewData.BreakID_B_side','SideA.ViewData.Status','SideA.ViewData.BreakID_A_side','SideA.ViewData._ID','SideB.ViewData._ID','SideB.ViewData.Side0_UniqueIds','SideA.ViewData.Side1_UniqueIds'],1))[:, 2]

X_test['Predicted_action'] = rf_predictions
X_test['probability_No_pair'] = probability_class_0
X_test['probability_UMB'] = probability_class_1
X_test['probability_UMR'] = probability_class_2
X_test['Predicted_action'].value_counts()

# ## Two Step Modeling

X_test2 = test_file[model_cols_2]
X_test2 = X_test2.reset_index()
X_test2 = X_test2.drop('index',1)
X_test2 = X_test2.fillna(0)

X_test2.shape
X_test2 = X_test2.drop_duplicates()
X_test2 = X_test2.reset_index()
X_test2 = X_test2.drop('index',1)

X_test2.shape

## TODO Import MOdel2 as per the two step modelling process

filename2 = 'OakTree_final_model2_step_two.sav'
clf2 = pickle.load(open(filename2, 'rb'))

# Actual class predictions
rf_predictions2 = clf2.predict(X_test2.drop(['SideB.ViewData.Status','SideB.ViewData.BreakID_B_side', 'SideA.ViewData.Status','SideA.ViewData.BreakID_A_side','SideA.ViewData._ID','SideB.ViewData._ID','SideB.ViewData.Side0_UniqueIds','SideA.ViewData.Side1_UniqueIds'],1))

# Probabilities for each class
rf_probs2 = clf2.predict_proba(X_test2.drop(['SideB.ViewData.Status','SideB.ViewData.BreakID_B_side', 'SideA.ViewData.Status','SideA.ViewData.BreakID_A_side','SideA.ViewData._ID','SideB.ViewData._ID','SideB.ViewData.Side0_UniqueIds','SideA.ViewData.Side1_UniqueIds'],1))[:, 1]

probability_class_0_two = clf2.predict_proba(X_test2.drop(['SideB.ViewData.Status','SideB.ViewData.BreakID_B_side','SideA.ViewData.Status','SideA.ViewData.BreakID_A_side','SideA.ViewData._ID','SideB.ViewData._ID','SideB.ViewData.Side0_UniqueIds','SideA.ViewData.Side1_UniqueIds'],1))[:, 0]
probability_class_1_two = clf2.predict_proba(X_test2.drop(['SideB.ViewData.Status','SideB.ViewData.BreakID_B_side', 'SideA.ViewData.Status','SideA.ViewData.BreakID_A_side','SideA.ViewData._ID','SideB.ViewData._ID','SideB.ViewData.Side0_UniqueIds','SideA.ViewData.Side1_UniqueIds'],1))[:, 1]

X_test2['Predicted_action_2'] = rf_predictions2
X_test2['probability_No_pair_2'] = probability_class_0_two
X_test2['probability_UMB_2'] = probability_class_1_two

X_test2['Predicted_action_2'].value_counts()

X_test.shape

X_test = pd.concat([X_test, X_test2[['Predicted_action_2','probability_No_pair_2','probability_UMB_2']]],axis=1)

# ## New Aggregation
X_test['Tolerance_level'] = np.abs(X_test['probability_UMB_2'] - X_test['probability_No_pair_2'])
b_side_agg = X_test.groupby(['SideB.ViewData.Side0_UniqueIds'])['Predicted_action_2'].unique().reset_index()
a_side_agg = X_test.groupby(['SideA.ViewData.Side1_UniqueIds'])['Predicted_action_2'].unique().reset_index()


# ## UMR segregation
umr_ids_0 = umr_seg(X_test)



# ## 1st Prediction Table for One to One UMR

final_umr_table = X_test[X_test['SideB.ViewData.Side0_UniqueIds'].isin(umr_ids_0) & (X_test['Predicted_action']=='UMR_One_to_One')]
final_umr_table = final_umr_table[['SideB.ViewData.Side0_UniqueIds','SideA.ViewData.Side1_UniqueIds','SideB.ViewData.BreakID_B_side','SideA.ViewData.BreakID_A_side','Predicted_action','probability_No_pair','probability_UMB','probability_UMR']]

# ## No-Pair segregation

no_pair_ids_b_side, no_pair_ids_a_side = no_pair_seg(X_test)

X_test[(X_test['SideA.ViewData.Side1_UniqueIds'].isin(no_pair_ids_a_side))]['Predicted_action_2'].value_counts()

X_test.groupby(['SideA.ViewData.Side1_UniqueIds'])['Predicted_action_2'].unique().reset_index()

X_test[X_test['SideA.ViewData.Side1_UniqueIds'].isin(no_pair_ids_a_side)]['Predicted_action_2'].value_counts()

final_open_table = X_test[(X_test['SideB.ViewData.Side0_UniqueIds'].isin(no_pair_ids_b_side)) | (X_test['SideA.ViewData.Side1_UniqueIds'].isin(no_pair_ids_a_side))]

final_open_table = final_open_table[['SideB.ViewData.Side0_UniqueIds','SideA.ViewData.Side1_UniqueIds','SideB.ViewData.BreakID_B_side','SideA.ViewData.BreakID_A_side','Predicted_action_2','probability_No_pair_2','probability_UMB_2','probability_UMR']]

final_open_table['probability_UMR'] = 0.00010
final_open_table = final_open_table.rename(columns = {'Predicted_action_2':'Predicted_action','probability_No_pair_2':'probability_No_pair','probability_UMB_2':'probability_UMB'})


b_side_open_table = final_open_table.groupby('SideB.ViewData.Side0_UniqueIds')[['probability_No_pair','probability_UMB','probability_UMR']].mean().reset_index()
a_side_open_table = final_open_table.groupby('SideA.ViewData.Side1_UniqueIds')[['probability_No_pair','probability_UMB','probability_UMR']].mean().reset_index()

a_side_open_table = a_side_open_table[a_side_open_table['SideA.ViewData.Side1_UniqueIds'].isin(no_pair_ids_a_side)]
b_side_open_table = b_side_open_table[b_side_open_table['SideB.ViewData.Side0_UniqueIds'].isin(no_pair_ids_b_side)]

b_side_open_table = b_side_open_table.reset_index().drop('index',1)
a_side_open_table = a_side_open_table.reset_index().drop('index',1)

final_no_pair_table = pd.concat([a_side_open_table,b_side_open_table], axis=0)
final_no_pair_table = final_no_pair_table.reset_index().drop('index',1)
#
#final_no_pair_table = pd.merge(final_no_pair_table, final_open_table[['SideA.ViewData.Side1_UniqueIds','SideA.ViewData.BreakID_A_side']].drop_duplicates(), on = 'SideA.ViewData.Side1_UniqueIds', how='left')
#final_no_pair_table = pd.merge(final_no_pair_table, final_open_table[['SideB.ViewData.Side0_UniqueIds','SideB.ViewData.BreakID_B_side']].drop_duplicates(), on = 'SideB.ViewData.Side0_UniqueIds', how='left')
#

final_no_pair_table = normalize_final_no_pair_table_col_names(fun_final_no_pair_table = final_no_pair_table)
final_no_pair_table_copy = final_no_pair_table.copy()

final_no_pair_table_copy['ViewData.Side0_UniqueIds'] = final_no_pair_table_copy['ViewData.Side0_UniqueIds'].astype(str)
final_no_pair_table_copy['ViewData.Side1_UniqueIds'] = final_no_pair_table_copy['ViewData.Side1_UniqueIds'].astype(str)
 
final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.Side0_UniqueIds']=='None','Side0_1_UniqueIds'] = final_no_pair_table_copy['ViewData.Side1_UniqueIds']
final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.Side1_UniqueIds']=='None','Side0_1_UniqueIds'] = final_no_pair_table_copy['ViewData.Side0_UniqueIds']

final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.Side0_UniqueIds']=='nan','Side0_1_UniqueIds'] = final_no_pair_table_copy['ViewData.Side1_UniqueIds']
final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.Side1_UniqueIds']=='nan','Side0_1_UniqueIds'] = final_no_pair_table_copy['ViewData.Side0_UniqueIds']

del final_no_pair_table_copy['ViewData.Side0_UniqueIds']
del final_no_pair_table_copy['ViewData.Side1_UniqueIds']


final_no_pair_table_copy = final_no_pair_table_copy.append(no_pair_ids_df)

final_no_pair_table_copy = pd.merge(final_no_pair_table_copy, meo_df[['ViewData.Side1_UniqueIds','ViewData.BreakID','ViewData.Task ID','ViewData.Task Business Date','ViewData.Source Combination Code']].drop_duplicates(), left_on = 'Side0_1_UniqueIds',right_on = 'ViewData.Side1_UniqueIds', how='left')
final_no_pair_table_copy = pd.merge(final_no_pair_table_copy, meo_df[['ViewData.Side0_UniqueIds','ViewData.BreakID','ViewData.Task ID','ViewData.Task Business Date','ViewData.Source Combination Code']].drop_duplicates(), left_on = 'Side0_1_UniqueIds',right_on = 'ViewData.Side0_UniqueIds', how='left')
#    #no_pair_ids_df = no_pair_ids_df.rename(columns={'0':'filter_key'})
final_no_pair_table_copy['Predicted_Status'] = 'OB'
final_no_pair_table_copy['Predicted_action'] = 'No-Pair'
final_no_pair_table_copy['ML_flag'] = 'ML'
final_no_pair_table_copy['SetupID'] = setup_code 

final_no_pair_table_copy['ViewData.Task ID_x'] = final_no_pair_table_copy['ViewData.Task ID_x'].astype(str)
final_no_pair_table_copy['ViewData.Task ID_y'] = final_no_pair_table_copy['ViewData.Task ID_y'].astype(str)
 
final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.Task ID_x']=='None','Task ID'] = final_no_pair_table_copy['ViewData.Task ID_y']
final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.Task ID_y']=='None','Task ID'] = final_no_pair_table_copy['ViewData.Task ID_x']

final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.Task ID_x']=='nan','Task ID'] = final_no_pair_table_copy['ViewData.Task ID_y']
final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.Task ID_y']=='nan','Task ID'] = final_no_pair_table_copy['ViewData.Task ID_x']


final_no_pair_table_copy['ViewData.BreakID_x'] = final_no_pair_table_copy['ViewData.BreakID_x'].astype(str)
final_no_pair_table_copy['ViewData.BreakID_y'] = final_no_pair_table_copy['ViewData.BreakID_y'].astype(str)
 
final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.BreakID_x']=='None','BreakID'] = final_no_pair_table_copy['ViewData.BreakID_y']
final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.BreakID_y']=='None','BreakID'] = final_no_pair_table_copy['ViewData.BreakID_x']

final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.BreakID_x']=='nan','BreakID'] = final_no_pair_table_copy['ViewData.BreakID_y']
final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.BreakID_y']=='nan','BreakID'] = final_no_pair_table_copy['ViewData.BreakID_x']

final_no_pair_table_copy['ViewData.Task Business Date_x'] = final_no_pair_table_copy['ViewData.Task Business Date_x'].astype(str)
final_no_pair_table_copy['ViewData.Task Business Date_y'] = final_no_pair_table_copy['ViewData.Task Business Date_y'].astype(str)
 
final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.Task Business Date_x']=='None','Task Business Date'] = final_no_pair_table_copy['ViewData.Task Business Date_y']
final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.Task Business Date_y']=='None','Task Business Date'] = final_no_pair_table_copy['ViewData.Task Business Date_x']

final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.Task Business Date_x']=='nan','Task Business Date'] = final_no_pair_table_copy['ViewData.Task Business Date_y']
final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.Task Business Date_y']=='nan','Task Business Date'] = final_no_pair_table_copy['ViewData.Task Business Date_x']

final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.Task Business Date_x']=='NaT','Task Business Date'] = final_no_pair_table_copy['ViewData.Task Business Date_y']
final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.Task Business Date_y']=='NaT','Task Business Date'] = final_no_pair_table_copy['ViewData.Task Business Date_x']

final_no_pair_table_copy['ViewData.Source Combination Code_x'] = final_no_pair_table_copy['ViewData.Source Combination Code_x'].astype(str)
final_no_pair_table_copy['ViewData.Source Combination Code_y'] = final_no_pair_table_copy['ViewData.Source Combination Code_y'].astype(str)
 
final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.Source Combination Code_x']=='None','Source Combination Code'] = final_no_pair_table_copy['ViewData.Source Combination Code_y']
final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.Source Combination Code_y']=='None','Source Combination Code'] = final_no_pair_table_copy['ViewData.Source Combination Code_x']

final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.Source Combination Code_x']=='nan','Source Combination Code'] = final_no_pair_table_copy['ViewData.Source Combination Code_y']
final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.Source Combination Code_y']=='nan','Source Combination Code'] = final_no_pair_table_copy['ViewData.Source Combination Code_x']


final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.Source Combination Code_x']=='NaT','Source Combination Code'] = final_no_pair_table_copy['ViewData.Source Combination Code_y']
final_no_pair_table_copy.loc[final_no_pair_table_copy['ViewData.Source Combination Code_y']=='NaT','Source Combination Code'] = final_no_pair_table_copy['ViewData.Source Combination Code_x']


final_no_pair_table_copy['Final_predicted_break'] = ''

filepaths_no_pair_table = '\\\\vitblrdevcons01\\Raman  Strategy ML 2.0\\All_Data\\OakTree\\UAT_Run\\X_Test_379\\final_no_pair_table.csv'
final_no_pair_table_copy.to_csv(filepaths_no_pair_table)

final_umr_table_copy = final_umr_table.copy()
final_umr_table_copy = normalize_final_no_pair_table_col_names(fun_final_no_pair_table = final_umr_table_copy)

#final_umr_table_copy['ViewData.Side0_UniqueIds'] = final_umr_table_copy['ViewData.Side0_UniqueIds'].astype(str)
#final_umr_table_copy['ViewData.Side1_UniqueIds'] = final_umr_table_copy['ViewData.Side1_UniqueIds'].astype(str)
# 
#final_umr_table_copy.loc[final_umr_table_copy['ViewData.Side0_UniqueIds']=='None','Side0_1_UniqueIds'] = final_umr_table_copy['ViewData.Side1_UniqueIds']
#final_umr_table_copy.loc[final_umr_table_copy['ViewData.Side1_UniqueIds']=='None','Side0_1_UniqueIds'] = final_umr_table_copy['ViewData.Side0_UniqueIds']
#
#final_umr_table_copy.loc[final_umr_table_copy['ViewData.Side0_UniqueIds']=='nan','Side0_1_UniqueIds'] = final_umr_table_copy['ViewData.Side1_UniqueIds']
#final_umr_table_copy.loc[final_umr_table_copy['ViewData.Side1_UniqueIds']=='nan','Side0_1_UniqueIds'] = final_umr_table_copy['ViewData.Side0_UniqueIds']

#del final_umr_table_copy['ViewData.Side0_UniqueIds']
#del final_umr_table_copy['ViewData.Side1_UniqueIds']

final_umr_table_copy = pd.merge(final_umr_table_copy, meo_df[['ViewData.Side1_UniqueIds','ViewData.Task ID','ViewData.Task Business Date','ViewData.Source Combination Code']].drop_duplicates(), on = 'ViewData.Side1_UniqueIds', how='left')
final_umr_table_copy['Predicted_Status'] = 'UMR'
#final_umr_table_copy['Predicted_action'] = 'No-Pair'
final_umr_table_copy['ML_flag'] = 'ML'
final_umr_table_copy['SetupID'] = setup_code 

filepaths_umr_table = '\\\\vitblrdevcons01\\Raman  Strategy ML 2.0\\All_Data\\OakTree\\UAT_Run\\X_Test_379\\final_umr_table.csv'
final_umr_table_copy.to_csv(filepaths_umr_table)

change_names_of_umr_table_mapping_dict = {
                                            'ViewData.Side0_UniqueIds' : 'Side0_UniqueIds',
                                            'ViewData.Side1_UniqueIds' : 'Side1_UniqueIds',
                                            'ViewData.BreakID_Side0' : 'BreakID',
                                            'ViewData.BreakID_Side1' : 'Final_predicted_break',
                                            'ViewData.Task ID' : 'Task ID',
                                            'ViewData.Task Business Date' : 'Task Business Date',
                                            'ViewData.Source Combination Code' : 'Source Combination Code'
                                        }

final_no_pair_table_copy['Task Business Date'] = pd.to_datetime(final_no_pair_table_copy['Task Business Date'])
final_no_pair_table_copy['Task Business Date'] = final_no_pair_table_copy['Task Business Date'].map(lambda x: dt.datetime.strftime(x, '%Y-%m-%dT%H:%M:%SZ'))
final_no_pair_table_copy['Task Business Date'] = pd.to_datetime(final_no_pair_table_copy['Task Business Date'])


final_umr_table_copy.rename(columns = change_names_of_umr_table_mapping_dict, inplace = True)

final_umr_table_copy['Task Business Date'] = pd.to_datetime(final_umr_table_copy['Task Business Date'])
final_umr_table_copy['Task Business Date'] = final_umr_table_copy['Task Business Date'].map(lambda x: dt.datetime.strftime(x, '%Y-%m-%dT%H:%M:%SZ'))
final_umr_table_copy['Task Business Date'] = pd.to_datetime(final_umr_table_copy['Task Business Date'])



final_no_pair_table_copy.rename(columns = change_names_of_umr_table_mapping_dict, inplace = True)
cols_for_database = list(final_umr_table_copy.columns)


final_no_pair_table_to_write = final_no_pair_table_copy[cols_for_database]

final_table_to_write = final_no_pair_table_to_write.append(final_umr_table_copy)

final_table_to_write['PredictedComment'] = ''

#Changing data types of columns as follows:
#Side0_UniqueIds, Side1_UniqueIds, Final_predicted_break, Predicted_action, probability_No_pair, probability_UMB, probability_UMR, BusinessDate, SourceCombinationCode, Predicted_Status, ML_flag - string
#BreakID, TaskID - int64
#SetupID - int32

final_table_to_write[['Side0_UniqueIds', 'Side1_UniqueIds', 'Final_predicted_break', 'Predicted_action', 'probability_No_pair', 'probability_UMB', 'probability_UMR', 'Source Combination Code', 'Predicted_Status', 'ML_flag']] = final_table_to_write[['Side0_UniqueIds', 'Side1_UniqueIds', 'Final_predicted_break', 'Predicted_action', 'probability_No_pair', 'probability_UMB', 'probability_UMR', 'Source Combination Code', 'Predicted_Status', 'ML_flag']].astype(str)

final_table_to_write[['BreakID', 'Task ID']] = final_table_to_write[['BreakID', 'Task ID']].astype(float)
final_table_to_write[['BreakID', 'Task ID']] = final_table_to_write[['BreakID', 'Task ID']].astype(np.int64)

final_table_to_write[['SetupID']] = final_table_to_write[['SetupID']].astype(int)

#final_table_to_write['Task ID'] = final_table_to_write['Task ID'].astype(float)
#final_table_to_write['Task ID'] = final_table_to_write['Task ID'].astype(np.int64)

change_col_names_dict = {
                        'Task ID' : 'TaskID',
                        'Task Business Date' : 'BusinessDate',
                        'Source Combination Code' : 'SourceCombinationCode'
                        }
final_table_to_write.rename(columns = change_col_names_dict, inplace = True)

filepaths_table_to_write = '\\\\vitblrdevcons01\\Raman  Strategy ML 2.0\\All_Data\\OakTree\\UAT_Run\\X_Test_379\\final_table_to_write.csv'

filepaths_final_no_pair_table_copy = '\\\\vitblrdevcons01\\Raman  Strategy ML 2.0\\All_Data\\OakTree\\final_no_pair_table.csv'
filepaths_meo_df = '\\\\vitblrdevcons01\\Raman  Strategy ML 2.0\\All_Data\\OakTree\\meo_df.csv'

final_no_pair_table.to_csv(filepaths_final_no_pair_table_copy)


final_no_pair_table_copy.to_csv(filepaths_final_no_pair_table_copy)
meo_df.to_csv(filepaths_meo_df)

#data_dict = final_table_to_write.to_dict("records")
#coll_1_for_writing_prediction_data = db_1_for_MEO_data['MLPrediction_Cash']
#coll_1_for_writing_prediction_data.insert_many(data_dict) 
#

#Commenting part starts here
os.chdir('D:/ViteosModel/Abhijeet - Comment')

final_no_pair_table_col_names_mapping_dict = {
                                  'Side0_UniqueIds' : 'ViewData.Side0_UniqueIds',
                                  'Side1_UniqueIds' : 'ViewData.Side1_UniqueIds',
                                    }
final_no_pair_table_copy.rename(columns = final_no_pair_table_col_names_mapping_dict, inplace = True)

brk = final_no_pair_table_copy[final_no_pair_table_copy['Predicted_action'] == 'No-Pair']

brk['Side0_UniqueIds'] = brk['Side0_UniqueIds'].fillna('AA')
brk['Side1_UniqueIds'] = brk['Side1_UniqueIds'].fillna('BB')

def fid1(a,b,c):
    if a=='OB':
        if b =='AA':
            return c
        else:
            return b
    else:
        return '12345'

brk['final_ID'] = brk.apply(lambda row : fid1(row['Predicted_Status'],row['Side0_UniqueIds'],row['Side1_UniqueIds']),axis =1 )

side0_id = list(set(brk[brk['Side1_UniqueIds'] =='BB']['Side0_UniqueIds']))
side1_id = list(set(brk[brk['Side0_UniqueIds'] =='AA']['Side1_UniqueIds']))

meo1 = meo_df[meo_df['ViewData.Side0_UniqueIds'].isin(side0_id)]
meo2 = meo_df[meo_df['ViewData.Side1_UniqueIds'].isin(side1_id)]

frames = [meo1, meo2]

df1 = pd.concat(frames)
df1 = df1.reset_index()
df1 = df1.drop('index', axis = 1)

df1 = df1.drop_duplicates()

df1['ViewData.Side0_UniqueIds'] = df1['ViewData.Side0_UniqueIds'].fillna('AA')
df1['ViewData.Side1_UniqueIds'] = df1['ViewData.Side1_UniqueIds'].fillna('BB')

def fid(a,b):
   
    if ( b=='BB'):
        return a
    else:
        return b

df1['final_ID'] = df1.apply(lambda row: fid(row['ViewData.Side0_UniqueIds'],row['ViewData.Side1_UniqueIds']),axis =1)

df1 = df1.sort_values(['final_ID','ViewData.Business Date'], ascending = [True, True])

uni2 = df1.groupby(['final_ID','ViewData.Task Business Date']).last().reset_index()

uni2 = uni2.sort_values(['final_ID','ViewData.Task Business Date'], ascending = [True, True])


# #### Trade date vs Settle date and future dated trade
df2 = uni2.copy()

df2['ViewData.Settle Date'] = pd.to_datetime(df2['ViewData.Settle Date'])
df2['ViewData.Trade Date'] = pd.to_datetime(df2['ViewData.Trade Date'])
df2['ViewData.Task Business Date'] = pd.to_datetime(df2['ViewData.Task Business Date'])

df2['ViewData.Task Business Date1'] = df2['ViewData.Task Business Date'].dt.date
df2['ViewData.Settle Date1'] = df2['ViewData.Settle Date'].dt.date
df2['ViewData.Trade Date1'] = df2['ViewData.Trade Date'].dt.date

df2['ViewData.SettlevsTrade Date'] = (df2['ViewData.Settle Date1'] - df2['ViewData.Trade Date1']).dt.days
df2['ViewData.SettlevsTask Date'] = (df2['ViewData.Task Business Date1'] - df2['ViewData.Settle Date1']).dt.days
df2['ViewData.TaskvsTrade Date'] = (df2['ViewData.Task Business Date1'] - df2['ViewData.Trade Date1']).dt.days

df = pd.read_excel('Mapping variables for variable cleaning.xlsx', sheet_name='General')

def make_dict(row):
    keys_l = str(row['Keys']).lower()
    keys_s = keys_l.split(', ')
    keys = tuple(keys_s)
    return keys

df['tuple'] = df.apply(make_dict, axis=1)

clean_map_dict = df.set_index('tuple')['Value'].to_dict()

df2['ViewData.Transaction Type'] = df2['ViewData.Transaction Type'].apply(lambda x : x.lower() if type(x)==str else x)
df2['ViewData.Asset Type Category'] = df2['ViewData.Asset Type Category'].apply(lambda x : x.lower() if type(x)==str else x)
df2['ViewData.Investment Type'] = df2['ViewData.Investment Type'].apply(lambda x : x.lower() if type(x)==str else x)
df2['ViewData.Prime Broker'] = df2['ViewData.Prime Broker'].apply(lambda x : x.lower() if type(x)==str else x)

def clean_mapping(item):
    item1 = item.split()
    
    
    ttype = []
    
    
    for x in item1:
        ttype1 = []
        for key, value in clean_map_dict.items():
            
    
        
        
            if x in key:
                a = value
                ttype1.append(a)
           
        if len(ttype1)==0:
            ttype1.append(x)
        ttype = ttype + ttype1
        
    return ' '.join(ttype)

df2['ViewData.Transaction Type1'] = df2['ViewData.Transaction Type'].apply(lambda x : clean_mapping(x) if type(x)==str else x)
df2['ViewData.Asset Type Category1'] = df2['ViewData.Asset Type Category'].apply(lambda x : clean_mapping(x) if type(x)==str else x)
df2['ViewData.Investment Type1'] = df2['ViewData.Investment Type'].apply(lambda x : clean_mapping(x) if type(x)==str else x)
df2['ViewData.Prime Broker1'] = df2['ViewData.Prime Broker'].apply(lambda x : clean_mapping(x) if type(x)==str else x)

def is_num(item):
    try:
        float(item)
        return True
    except ValueError:
        return False

def is_date_format(item):
    try:
        parse(item, fuzzy=False)
        return True
    
    except ValueError:
        return False
    
def date_edge_cases(item):
    if len(item) == 5 and item[2] =='/' and is_num(item[:2]) and is_num(item[3:]):
        return True
    return False

def comb_clean(x):
    k = []
    for item in x.split():
        if ((is_num(item)==False) and (is_date_format(item)==False) and (date_edge_cases(item)==False)):
            k.append(item)
    return ' '.join(k)

df2['ViewData.Transaction Type1'] = df2['ViewData.Transaction Type1'].apply(lambda x : comb_clean(x) if type(x)==str else x)
df2['ViewData.Asset Type Category1'] = df2['ViewData.Asset Type Category1'].apply(lambda x : comb_clean(x) if type(x)==str else x)
df2['ViewData.Investment Type1'] = df2['ViewData.Investment Type1'].apply(lambda x : comb_clean(x) if type(x)==str else x)
df2['ViewData.Prime Broker1'] = df2['ViewData.Prime Broker1'].apply(lambda x : comb_clean(x) if type(x)==str else x)

df2['ViewData.Transaction Type1'] = df2['ViewData.Transaction Type1'].apply(lambda x : 'paydown' if x=='pay down' else x)

trade_types = ['buy','sell','cover short', 'sell short', 'forward', 'forwardfx', 'spotfx']

dfkk = df2[df2['ViewData.Transaction Type1'].isin(trade_types)]

dffk2 = dfkk[dfkk['ViewData.Side0_UniqueIds']=='AA']
dffk3 = dfkk[dfkk['ViewData.Side1_UniqueIds']=='BB']


# #### Geneva side

def canceltrade(x,y):
    if x =='buy' and y>0:
        k = 1
    elif x =='sell' and y<0:
        k = 1
    else:
        k = 0
    return k

dffk3['cancel_marker'] = dffk3.apply(lambda x : canceltrade(x['ViewData.Transaction Type1'],x['ViewData.Accounting Net Amount']), axis = 1)

dffk3[dffk3['cancel_marker'] == 1]['ViewData.Transaction ID'].nunique()

cancel_trade = list(set(dffk3[dffk3['cancel_marker'] == 1]['ViewData.Transaction ID']))

# - Segegation of canceltrade with comment

original = dffk3[dffk3['ViewData.Transaction ID'].isin(cancel_trade)]

original['predicted category'] = 'Original of Cancelled trade'

def cancelcomment(x,y):
    com1 = 'This is original of cancelled trade with tran id'
    com2 = 'on settle date'
    com = com1 + ' ' +  str(x) + ' ' + com2 + str(y)
    return com

#TODO : Look into how to resolve empty created dataframes :  Below code line gave error due to empty dataframe named original
original['predicted comment'] = original.apply(lambda x : cancelcomment(x['ViewData.Transaction ID'],x['ViewData.Settle Date1']), axis = 1)

cancellation = dffk3[dffk3['cancel_marker'] == 1]

cancellation['predicted category'] = 'Cancelled trade'

def cancelcomment1(x,y):
    com1 = 'This is cancelled trade with tran id'
    com2 = 'on settle date'
    com = com1 + ' ' +  str(x) + ' ' + com2 + str(y)
    return com

#TODO : Look into how to resolve empty created dataframes :  Below code line gave error due to empty dataframe named cancellation
cancellation['predicted comment'] =  cancellation.apply(lambda x : cancelcomment1(x['ViewData.Transaction ID'],x['ViewData.Settle Date1']), axis = 1)

cancel_fin = pd.concat([original,cancellation])

sel_col_1 = ['final_ID','predicted category','predicted comment']

#TODO : Look into how to resolve empty created dataframes :  Below code line gave error due to empty dataframe named cancel_fin
cancel_fin = cancel_fin[sel_col_1]
cancel_fin.to_csv('Comment file oaktree 2 sep testing p1.csv')


# - Moving Forward
dffk3 = dffk3[~dffk3['ViewData.Transaction ID'].isin(cancel_trade)]
# #### Broker side

dffk2['cancel_marker'] = dffk2.apply(lambda x : canceltrade(x['ViewData.Transaction Type1'],x['ViewData.Cust Net Amount']), axis = 1)

#TODO : Note that dummy dataframe below has 0 rows
dummy = dffk2[dffk2['cancel_marker']==1]

dummy1 = dffk2[dffk2['cancel_marker']!=1]

pool =[]
key_index =[]
training_df =[]
call1 = []

appended_data = []

no_pair_ids = []
#max_rows = 5

k = list(set(list(set(dummy['ViewData.Task Business Date1']))))
k1 = k

for d in tqdm(k1):
    aa1 = dummy[dummy['ViewData.Task Business Date1']==d]
    bb1 = dummy1[dummy1['ViewData.Task Business Date1']==d]
    aa1['marker'] = 1
    bb1['marker'] = 1
    
    aa1 = aa1.reset_index()
    aa1 = aa1.drop('index',1)
    bb1 = bb1.reset_index()
    bb1 = bb1.drop('index', 1)
    print(aa1.shape)
    print(bb1.shape)
    
    aa1.columns = ['SideB.' + x  for x in aa1.columns] 
    bb1.columns = ['SideA.' + x  for x in bb1.columns]
    
    cc1 = pd.merge(aa1,bb1, left_on = 'SideB.marker', right_on = 'SideA.marker', how = 'outer')
    appended_data.append(cc1)
    print(cc1.shape)

#TODO : appended_data is empty list. So we get error for below line of code
cancel_broker = pd.concat(appended_data)


def amountelim(row):
   
   
   
    if (row['SideA.ViewData.Mapped Custodian Account'] == row['SideB.ViewData.Mapped Custodian Account']):
        a = 1
    else:
        a = 0
        
    if ((row['SideB.ViewData.Cust Net Amount']) == -(row['SideA.ViewData.Cust Net Amount'])):
        b = 1
    else:
        b = 0
    
    if (row['SideA.ViewData.Fund'] == row['SideB.ViewData.Fund']):
        c = 1
    else:
        c = 0
        
    if (row['SideA.ViewData.Currency'] == row['SideB.ViewData.Currency']):
        d = 1
    else:
        d = 0
    
    if (row['SideA.ViewData.Settle Date1'] == row['SideB.ViewData.Settle Date1']):
        e = 1
    else:
        e = 0
        
    if (row['SideA.ViewData.Transaction Type1'] == row['SideB.ViewData.Transaction Type1']):
        f = 1
    else:
        f = 0
        
    if (row['SideB.ViewData.Quantity'] == row['SideA.ViewData.Quantity']):
        g = 1
    else:
        g = 0
        
    if (row['SideB.ViewData.ISIN'] == row['SideA.ViewData.ISIN']):
        h = 1
    else:
        h = 0
        
    if (row['SideB.ViewData.CUSIP'] == row['SideA.ViewData.CUSIP']):
        i = 1
    else:
        i = 0
        
    if (row['SideB.ViewData.Ticker'] == row['SideA.ViewData.Ticker']):
        j = 1
    else:
        j = 0
        
    if (row['SideB.ViewData.Investment ID'] == row['SideA.ViewData.Investment ID']):
        k = 1
    else:
        k = 0
        
    return a, b, c ,d, e,f,g,h,i,j,k

#TODO : Below 3 lines of code gives error as cancel_broker is not defined as appended_data was empty
cancel_broker[['map_match','amt_match','fund_match','curr_match','sd_match','ttype_match','Qnt_match','isin_match','cusip_match','ticker_match','Invest_id']] = cancel_broker.apply(lambda row : amountelim(row), axis = 1,result_type="expand")

cancel_broker['key_match_sum'] = cancel_broker['map_match'] + cancel_broker['sd_match'] + cancel_broker['curr_match'] + cancel_broker['ttype_match'] + cancel_broker['Qnt_match'] + + cancel_broker['isin_match'] + + cancel_broker['cusip_match']

elim1 = cancel_broker[(cancel_broker['map_match']==1) & (cancel_broker['curr_match']==1)  & ((cancel_broker['isin_match']==1) |(cancel_broker['cusip_match']==1)| (cancel_broker['ticker_match']==1) | (cancel_broker['Invest_id']==1))]

id_listB = list(set(dummy['final_ID'])) 

#TODO : Below line of code gives error as elim1 is not defined as cancel_broker is not defined as appended_data was empty
id_listA = list(set(elim1['SideA.final_ID']))

c1 = dummy

#TODO : Below line of code gives error as idlist! is not defined as elim1 is not defined as cancel_broker is not defined as appended_data was empty

c2 = dummy1[dummy1['final_ID'].isin(id_listA)]

c1['predicted category'] = 'Original of Cancelled trade'
c1['predicted category'] = 'Original of Cancelled trade'

def cancelcomment(y):
    com1 = 'This is original of cancelled trade'
    com2 = 'on settle date'
    com = com1 + ' '  + com2 +' ' + str(y)
    return com

def cancelcomment1(y):
    com1 = 'This is cancelled trade with tran id'
    com2 = 'on settle date'
    com = com1 + ' ' + com2 + ' ' + str(y)
    return com

#TODO : Below 2 lines of code give error due to empty c1 and c2
c1['predicted comment'] =  c1.apply(lambda x : cancelcomment(x['ViewData.Settle Date1']))

c2['predicted comment'] = c2.apply(lambda x : cancelcomment(x['ViewData.Settle Date1']))

cancel_fin = pd.concat([c1,c2])

sel_col_1 = ['final_ID','predicted category','predicted comment']

#TODO : Below line of code give error due to empty cancel_fin
cancel_fin = cancel_fin[sel_col_1]
cancel_fin.to_csv('Comment file oaktree 2 sep testing p2.csv')


# - Moving ahead with transactions

#TODO : Below line of code give error due to empty id_listA
cancel_trade2 = id_listB+ id_listA

#TODO : Below line of code give error due to empty cancel_trade2
dffk2 = dffk2[~dffk2['final_ID'].isin(cancel_trade2)]

# #### Finding Pairs in Up and down

sel_col = ['ViewData.Currency', 
       'ViewData.Accounting Net Amount', 'ViewData.Age', 'ViewData.Asset Type Category1',
       
        'ViewData.Cust Net Amount',
       'ViewData.BreakID', 'ViewData.Business Date', 'ViewData.Cancel Amount',
       'ViewData.Cancel Flag', 'ViewData.ClusterID', 'ViewData.Commission',
       'ViewData.CUSIP',  
       'ViewData.Description',  'ViewData.Fund',
        'ViewData.Has Attachments',
       'ViewData.InternalComment1', 'ViewData.InternalComment2',
       'ViewData.InternalComment3', 'ViewData.Investment ID',
       'ViewData.Investment Type1', 
       'ViewData.ISIN', 'ViewData.Keys', 
       'ViewData.Mapped Custodian Account', 
       
        'ViewData.Portfolio ID',
       'ViewData.Portolio', 'ViewData.Price', 'ViewData.Prime Broker1',
        
       'ViewData.Quantity',  'ViewData.Rule And Key',
       'ViewData.SEDOL', 'ViewData.Settle Date', 
       'ViewData.Status', 'ViewData.Strategy', 'ViewData.System Comments',
       'ViewData.Ticker', 'ViewData.Trade Date', 'ViewData.Trade Expenses',
       'ViewData.Transaction ID',
       'ViewData.Transaction Type1', 'ViewData.Underlying Cusip',
       'ViewData.Underlying Investment ID', 'ViewData.Underlying ISIN',
       'ViewData.Underlying Sedol', 'ViewData.Underlying Ticker',
       'ViewData.UserTran1', 'ViewData.UserTran2', 
       'ViewData.Side0_UniqueIds', 'ViewData.Side1_UniqueIds',
       'ViewData.Task Business Date', 'final_ID',
        'ViewData.Task Business Date1',
       'ViewData.Settle Date1', 'ViewData.Trade Date1',
       'ViewData.SettlevsTrade Date', 'ViewData.SettlevsTask Date',
       'ViewData.TaskvsTrade Date'
      ]

dff4 = dffk2[sel_col]
dff5 = dffk3[sel_col]

# #### M cross N code
###################### loop 3 ###############################

pool =[]
key_index =[]
training_df =[]
call1 = []

appended_data = []

no_pair_ids = []
#max_rows = 5

k = list(set(list(set(dff5['ViewData.Task Business Date1'])) + list(set(dff4['ViewData.Task Business Date1']))))
k1 = k

for d in tqdm(k1):
    aa1 = dff4[dff4['ViewData.Task Business Date1']==d]
    bb1 = dff5[dff5['ViewData.Task Business Date1']==d]
    aa1['marker'] = 1
    bb1['marker'] = 1
    
    aa1 = aa1.reset_index()
    aa1 = aa1.drop('index',1)
    bb1 = bb1.reset_index()
    bb1 = bb1.drop('index', 1)
    print(aa1.shape)
    print(bb1.shape)
    
    aa1.columns = ['SideB.' + x  for x in aa1.columns] 
    bb1.columns = ['SideA.' + x  for x in bb1.columns]
    
    cc1 = pd.merge(aa1,bb1, left_on = 'SideB.marker', right_on = 'SideA.marker', how = 'outer')
    appended_data.append(cc1)
    print(cc1.shape)

df_213_1 = pd.concat(appended_data)

# ### Elimination Round 1 : Important key with Amount
#TODO : We just created df_213_1 above. If we rewrite on it, whats the point of creating it earlier? Ask Abhijeet. Also cancel_broker was empty, so below line also gave error
df_213_1 = cancel_broker.copy()

def amountelim(row):
   
   
   
    if (row['SideA.ViewData.Mapped Custodian Account'] == row['SideB.ViewData.Mapped Custodian Account']):
        a = 1
    else:
        a = 0
        
    if (row['SideB.ViewData.Cust Net Amount'] == row['SideA.ViewData.Accounting Net Amount']):
        b = 1
    else:
        b = 0
    
    if (row['SideA.ViewData.Fund'] == row['SideB.ViewData.Fund']):
        c = 1
    else:
        c = 0
        
    if (row['SideA.ViewData.Currency'] == row['SideB.ViewData.Currency']):
        d = 1
    else:
        d = 0
    
    if (row['SideA.ViewData.Settle Date1'] == row['SideB.ViewData.Settle Date1']):
        e = 1
    else:
        e = 0
        
    if (row['SideA.ViewData.Transaction Type1'] == row['SideB.ViewData.Transaction Type1']):
        f = 1
    else:
        f = 0
        
    return a, b, c ,d, e,f
    
df_213_1[['map_match','amt_match','fund_match','curr_match','sd_match','ttype_match']] = df_213_1.apply(lambda row : amountelim(row), axis = 1,result_type="expand")

df_213_1['key_match_sum'] = df_213_1['map_match'] + df_213_1['sd_match'] + df_213_1['curr_match']

# #### Inclusion of the fuzz matches

# #### Inclusion of the fuzz matches

# In[82]:


#ttype_soros = pd.read_csv('Soros for preprocess1/inzi training file/UMR ttype oaktree.csv')


# In[83]:


#ttype_combo_soros = list(ttype_soros['combined'])


# In[330]:

def ttype_match(a,b):
    if ((type(a)==float) | (type(b)==float)):
        if a==b:
            return 1
        else:
            return 0
    else:
        a = a.lower()
        b = b.lower()
        if a==b:
            return 1
        else:
            d = str(a)+str(b)
            e = str(b)+str(a)
        #print(d)
            if ((d in ttype_combo_soros) | (e in ttype_combo_soros)):
                return 1
            else:
                return 0
            return 0

#TODO : Below line of code gives following error
#        if ((d in ttype_combo_soros) | (e in ttype_combo_soros)):
#    NameError: ("name 'ttype_combo_soros' is not defined", 'occurred at index 50') for below code
#NOTE : We have commented out ttype_combo_soros df. Do we have to do it? Also, if we have to do it, then get the file from Abhijeet in line 2226

df_213_1['ttype_match_bool'] = df_213_1.apply(lambda x : ttype_match(x['SideA.ViewData.Transaction Type1'],x['SideB.ViewData.Transaction Type1']), axis = 1)

#TODO : Get this file from Abhijeet
itype_soros = pd.read_csv('UMR Itype oaktree.csv')

itype_combo_soros = list(set(itype_soros['combined']))

a = []
for item in itype_combo_soros:
    if type(item) == float:
        item = item
    else:
        item = item.lower()
    a.append(item)
    
def itype_match(a,b):
    if ((type(a)==float) | (type(b)==float)):
        if a==b:
            return 1
        else:
            return 0
    else:
        a = a.lower()
        b = b.lower()
        if a==b:
            return 1
        else:
            d = str(a)+str(b)
            e = str(b)+str(a)
        #print(d)
            if ((d in itype_combo_soros) | (e in itype_combo_soros)):
                return 1
            else:
                return 0

#TODO : df_213_1 does not have column SideA.ViewData.Investment Type, but SideA.ViewData.Investment Type1. Do we have to use the latter, cause otherwise it gives an error
#NOTE : I changed to latter and moved ahead
df_213_1['itype_match_bool'] = df_213_1.apply(lambda x : itype_match(x['SideA.ViewData.Investment Type'],x['SideB.ViewData.Investment Type']), axis = 1)

asset_soros = pd.read_csv('UMR Asset oaktree.csv')

asset_combo_soros = list(set(asset_soros['combined']))

def asset_match(a,b):
    if ((type(a)==float) | (type(b)==float)):
        if a==b:
            return 1
        else:
            return 0
    else:
        a = a.lower()
        b = b.lower()
        if a==b:
            return 1
        else:
            d = str(a)+str(b)
            e = str(b)+str(a)
        #print(d)
            if ((d in asset_combo_soros) | (e in asset_combo_soros)):
                return 1
            else:
                return 0

#TODO : df_213_1 does not have column SideA.ViewData.Asset Type Category, but SideA.ViewData.Asset Type Category1. Do we have to use the latter, cause otherwise it gives an error
#NOTE : I changed to latter and moved ahead

df_213_1['asset_match_bool'] = df_213_1.apply(lambda x : asset_match(x['SideA.ViewData.Asset Type Category'],x['SideB.ViewData.Asset Type Category']), axis = 1)

#TODO : NOTE : elim1 has 0 rows.
elim1 = df_213_1[(df_213_1['amt_match']==1) & (df_213_1['key_match_sum']>=2)]

# - putting updown comments
def updownat(a,b,c,d,e):
    if a == 0:
        k = 'mapped custodian account'
    elif b==0:
        k = 'currency'
    elif c ==0 :
        k = 'Settle Date'
        
    elif d == 0:
        k = 'Investment type'
    else :
        k = 'transaction type'
        
    com = 'up/down at'+ ' ' + k
    return com

#TODO : NOTE : cancel, frames, c1, c2, id_listA and id_listB are all empty lists as elim1 has 0 rows.
id_listB = list(set(elim1['SideB.final_ID'])) 
id_listA = list(set(elim1['SideA.final_ID']))

c1 = dffk2[dffk2['final_ID'].isin(id_listB)]
c2 = dffk3[dffk3['final_ID'].isin(id_listA)]

c1['predicted category'] = 'updown'
c2['predicted category'] = 'updown'

c1['predicted comment'] = 'updown'
c2['predicted category'] = 'updown'

frames = [c1,c2]

cancel = pd.concat(frames)

cancel.to_csv('Comment file oaktree 2 sep testing p3.csv')

#TODO : NOTE : id_listA and id_listB are all empty lists as elim1 has 0 rows.

id_listB = list(set(elim1['SideB.final_ID'])) 
id_listA = list(set(elim1['SideA.final_ID']))

df_213_1 = df_213_1[~df_213_1['SideB.final_ID'].isin(id_listB)]
df_213_1 = df_213_1[~df_213_1['SideA.final_ID'].isin(id_listA)]

# #### Round 2 Elimination : Opposite Sign

def opposignelim(row):
   
    if (abs(row['SideB.ViewData.Cust Net Amount']) == abs(row['SideA.ViewData.Accounting Net Amount'])):
        b = 1
    else:
        b = 0
    return b

#
## In[524]:
#
#
#df_213_1['amt_abs_match'] = df_213_1.apply(lambda row : opposignelim(row), axis = 1)
#
#
## In[526]:
#
#
#dummy = df_213_1[(df_213_1['amt_abs_match']==1) & (df_213_1['key_match_sum']>=2) & (df_213_1['ttype_match']==1)]
#
#
## In[527]:
#
#
#
#
## In[177]:
#
#
#dummy = df_213_1[(df_213_1['amt_abs_match']==1) & (df_213_1['key_match_sum']>=2)]
#
#
## In[178]:
#
#
#
def itype_match(a,b):
    if ((type(a)==float) | (type(b)==float)):
        if a==b:
            return 1
        else:
            return 0
    else:
        a = a.lower()
        b = b.lower()
        if a==b:
            return 1
        else:
            d = str(a)+str(b)
            e = str(b)+str(a)
        #print(d)
            if ((d in a) | (e in a)):
                return 1
            else:
                return 0

#dummy['new_itype_match'] = dummy.apply(lambda x : itype_match(x['SideA.ViewData.Investment Type'],x['SideB.ViewData.Investment Type']), axis = 1)
#dummy = df_213_1[(df_213_1['amt_elim']==1)]
#id_listB = list(set(dummy['SideB.final_ID'])) 
#id_listA = list(set(dummy['SideA.final_ID']))

df_213_1 = df_213_1[~df_213_1['SideB.final_ID'].isin(id_listB)]
df_213_1 = df_213_1[~df_213_1['SideA.final_ID'].isin(id_listA)]

id_listB = list(set(df_213_1['SideB.final_ID'])) 
id_listA = list(set(df_213_1['SideA.final_ID']))

dff4 = dff4[dff4['final_ID'].isin(id_listB)]
dff5 = dff5[dff5['final_ID'].isin(id_listA)]

# ### Comment Categorisation using model
data = dff4.append(dff5)

#TODO : Below line of code has error. 'data' does not have new_pb1 column below. 
data['new_pb2'] = data.apply(lambda x : 'Geneva' if x['ViewData.Side0_UniqueIds'] != 'AA' else x['new_pb1'], axis = 1)

Pre_final = [    
'ViewData.Side0_UniqueIds','ViewData.Side1_UniqueIds','ViewData.BreakID',
 'ViewData.Currency',
 'ViewData.Custodian',
 'ViewData.ISIN',
 'ViewData.Mapped Custodian Account',
 'ViewData.Net Amount Difference Absolute',
 'ViewData.Portolio',
 'ViewData.Settle Date',
 'ViewData.Trade Date',
 'ViewData.Transaction Type1',
 'new_desc_cat',
 'ViewData.Accounting Net Amount',
 'ViewData.Asset Type Category1',
 'ViewData.CUSIP',
 'ViewData.Commission',
 'ViewData.Fund',
 'ViewData.Investment ID',
 'ViewData.Investment Type1',
 'ViewData.Price',
 'ViewData.Prime Broker1',
 'ViewData.Quantity',
 'ViewData.InternalComment2', 'ViewData.Description','new_pb2','new_pb1',
  'Category'
 ]
#TODO : Below line of code has error : ['new_pb1', 'Category', 'ViewData.Custodian', 'new_desc_cat', 'new_pb2', 'ViewData.Net Amount Difference Absolute'] not in index
data = data[Pre_final]

df_mod1 = data.copy()

#TODO : Below line of code has error. 'data' does not have ViewData.Custodian column below. 
df_mod1['ViewData.Custodian'] = df_mod1['ViewData.Custodian'].fillna('AA')
df_mod1['ViewData.Portolio'] = df_mod1['ViewData.Portolio'].fillna('bb')
df_mod1['ViewData.Settle Date'] = df_mod1['ViewData.Settle Date'].fillna(0)
df_mod1['ViewData.Trade Date'] = df_mod1['ViewData.Trade Date'].fillna(0)
df_mod1['ViewData.Accounting Net Amount'] = df_mod1['ViewData.Accounting Net Amount'].fillna(0)
df_mod1['ViewData.Asset Type Category1'] = df_mod1['ViewData.Asset Type Category1'].fillna('CC')
df_mod1['ViewData.CUSIP'] = df_mod1['ViewData.CUSIP'].fillna('DD')
df_mod1['ViewData.Fund'] = df_mod1['ViewData.Fund'].fillna('EE')
df_mod1['ViewData.Investment ID'] = df_mod1['ViewData.Investment ID'].fillna('FF')
df_mod1['ViewData.Investment Type1'] = df_mod1['ViewData.Investment Type1'].fillna('GG')
#df_mod1['ViewData.Knowledge Date'] = df_mod1['ViewData.Knowledge Date'].fillna(0)
df_mod1['ViewData.Price'] = df_mod1['ViewData.Price'].fillna(0)
df_mod1['ViewData.Prime Broker1'] = df_mod1['ViewData.Prime Broker1'].fillna("HH")
df_mod1['ViewData.Quantity'] = df_mod1['ViewData.Quantity'].fillna(0)
#df_mod1['ViewData.Sec Fees'] = df_mod1['ViewData.Sec Fees'].fillna(0)
#df_mod1['ViewData.Strike Price'] = df_mod1['ViewData.Strike Price'].fillna(0)
df_mod1['ViewData.Commission'] = df_mod1['ViewData.Commission'].fillna(0)
df_mod1['ViewData.Transaction Type1'] = df_mod1['ViewData.Transaction Type1'].fillna('kk')
df_mod1['ViewData.ISIN'] = df_mod1['ViewData.ISIN'].fillna('mm')

#TODO : Below line of code has error. 'data' does not have new_desc_cat column below. 
df_mod1['new_desc_cat'] = df_mod1['new_desc_cat'].fillna('nn')

#TODO : Below line of code has error. 'data' does not have Category column below. 
df_mod1['Category'] = df_mod1['Category'].fillna('NA')

df_mod1['ViewData.Description'] = df_mod1['ViewData.Description'].fillna('nn')

def fid(a,b):
   
    if ( b=='BB'):
        return a
    else:
        return b

df_mod1['final_ID'] = df_mod1.apply(lambda row: fid(row['ViewData.Side0_UniqueIds'],row['ViewData.Side1_UniqueIds']),axis =1)
data2 = df_mod1.copy()


# #### Separate Prediction of the Trade and Non trade
data21 = data2.copy()
cols = [ 
 'ViewData.Transaction Type1',
 'ViewData.Asset Type Category1',
 'new_desc_cat',
 'ViewData.Investment Type1',
 'new_pb2','new_pb1'
 ]

#TODO : Below line of code has error :  KeyError: "['new_pb1', 'new_pb2', 'new_desc_cat'] not in index"
data211 = data21[cols]

filename = 'finalized_model_oaktree_trade_v2.sav'
clf = pickle.load(open(filename, 'rb'))

#TODO : Evertyhing below has error : NameError: name 'data211' is not defined
X_test = data211.copy()
# Actual class predictions
cb_predictions = clf.predict(X_test)#.astype(str)

# Probabilities for each class
#cb_probs = clf.predict_proba(X_test)[:, 1]


# #### Testing of Model and final prediction file

# In[ ]:


#print(metrics.classification_report(y_test,cb_predictions, digits=3))


# In[ ]:


demo = []
for item in cb_predictions:
    demo.append(item[0])


# In[ ]:

#
#list1 = list(y_test)
#
#
## In[ ]:
#
#
#X_test1 = data21.drop(['Category'],axis = 1)
#y_test1 = data21['Category'] 
#
#
## In[ ]:
#

result = X_test.copy()


# In[ ]:


result = result.reset_index()


# In[ ]:


result['predicted'] = pd.Series(demo)
#result['Actual'] = pd.Series(list1)


# ### Code Ends Here

# In[134]:
