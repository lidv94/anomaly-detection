import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# def check_data(df):
#     data = df
#     print(data.shape)
#     all_data = list()
#     for col in list(data.columns):
#         try:         
#             cols = col
#             type_col = data[col].dtypes
#             per_null = round(((data[col].isnull().sum())*100)/data.shape[0],1)
#             sum_null = data[col].isnull().sum()
#             nunique = data[col].nunique()
            
#             if data[col].dtypes == object:
#                 remark = data[col].value_counts().index.tolist()[:5]
#             elif data[col].dtypes == int | data[col].dtypes == float:
#                 remark = [data[col].min(),data[col].mean(),data[col].max()]
#             elif data[col].dtypes == '<M8[ns]':
#                 remark = [data[col].min().dt.date,data[col].mean().dt.date,data[col].max().dt.date]    
#             # else:
#             #     remark = [data[col].min(),data[col].mean(),data[col].max()]
                
#             all_data.append([col,type_col,per_null,sum_null,nunique,remark])
#         except:
#             print(col)
    
#     df = pd.DataFrame(all_data,columns=['Columns','Type','%Null','#Null','#Unique','MinMeanMax_Unique'])
#     return df
# แก้ sub funtion เช่น int float ด้วยกัน -> เอารวมถ้าทำงานเหมือนกัน
# try except ออก

def check_data(df):
    data = df
    print(data.shape)
    all_data = list()
    for col in list(data.columns):
        try:         
            cols = col
            type_col = data[col].dtypes
            per_null = round(((data[col].isnull().sum())*100)/data.shape[0],1)
            sum_null = data[col].isnull().sum()
            nunique = data[col].nunique()
            
            if data[col].dtypes == object:
                remark = data[col].value_counts().index.tolist()[:5]
            elif data[col].dtypes == int:
                remark = [data[col].min(),data[col].mean(),data[col].max()]
            else:
                remark = [data[col].min(),data[col].mean(),data[col].max()]
                
            all_data.append([col,type_col,per_null,sum_null,nunique,remark])
        except:
            print(col)
    
    df = pd.DataFrame(all_data,columns=['Columns','Type','%Null','#Null','#Unique','MinMeanMax_Unique'])
    return df
 
def vis_eda(df:pd.core.frame.DataFrame,ncols=4,width=16,height=2):
    cols = df.columns
    n = len(cols)
    fig = plt.figure(figsize = (width,n*height))
    for i in range(n):
        plt.subplot(math.ceil(n/ncols), ncols, i+1)
        plt.title(cols[i])
        if df[cols[i]].dtypes == object:
            df[cols[i]].value_counts()[:10].plot.bar(rot=45)
            plt.xlabel('')
        elif df[cols[i]].dtypes == '<M8[ns]':
            df[cols[i]].dt.year.value_counts().sort_index().plot.line()
            plt.xlabel('')
        else:
            df[cols[i]].hist()

def join(df_left:pd.core.frame.DataFrame,df_right:pd.core.frame.DataFrame,how:str,left_on:list, right_on:list,key_notna:str):
    print('shape: left',df_left.shape,'/','right',df_right.shape)
    merged1 = df_left.merge(df_right,how=how,left_on=left_on, right_on=right_on)
    print('shape merged1:',merged1.shape)
    merged2 = merged1[merged1[key_notna].notna()]
    print('shape merged2:',merged2.shape)
    print('#missing transaction',merged1.shape[0]-merged2.shape[0],'/','%missing transaction',round(((merged1.shape[0]-merged2.shape[0])/merged1.shape[0])*100,2),'%')
    return merged1,merged2