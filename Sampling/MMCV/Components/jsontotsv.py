import csv
from mailbox import linesep
import json
import pandas as pd
import numpy as np
import sys
import os
import collections
from transformers import BasicTokenizer

def read_ssv(input_file, output,balanced=False,por=1):
        """Reads a tab separated value file."""
        df=pd.read_json(input_file, lines=True)
        a, b = np.split(df, [int(por*len(df))])
        df=a
        df = df.reset_index() 
        res = {}
        ress=[]
        dfs=pd.DataFrame(columns=['text','label'])
        print(df.shape[0])
        dfs.loc[0] = ['!LF!','!LF!']
        #dfs = dfs['text']
        for index,row in df.iterrows():
            t=row['text']
            l=row['label']
            res = list(zip(t, l))
            df1 = pd.DataFrame(res, columns =['text', 'label'])
            ress.append(df1)
            ress.append(dfs)
        df2=pd.concat(ress,ignore_index=True)
        df2.to_csv(output,sep=' ',index=False,header=False)
        print(df2.shape[0])

"""
input_dir = '/home/jon/Documentos/TFM/ecai2020-transformer_based_am-master/preprocessing/train.json'
output = '/home/jon/Documentos/TFM/ecai2020-transformer_based_am-master/preprocessing/train.tsv'
por=1
balanced=True
read_ssv(input_dir, output,por,balanced)
"""
"""
for i in range(3):
    input_dir = '/home/jon/Documentos/TFM/ecai2020-transformer_based_am-master/data/realconll/20shot/%s.json'%(i+1)
    output = '/home/jon/Documentos/TFM/ecai2020-transformer_based_am-master/data/realconll/20shot/%s/train.tsv'%(i+1)
    read_ssv(input_dir, output)
"""

input_dir = '/home/jon/Documentos/TFM/ecai2020-transformer_based_am-master/data/argument_components/manual_projections/deepl/awesome/train.json'
output = '/home/jon/Documentos/TFM/ecai2020-transformer_based_am-master/data/argument_components/manual_projections/deepl/awesome/40%/train.tsv'

read_ssv(input_dir, output,por=0.4)

    