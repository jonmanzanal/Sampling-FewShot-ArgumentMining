import pandas as pd
import numpy as np

def split(input,output,por,lower,balance):
    ig=[]
    ik=[]
    ih=[]
    it=[]
    df = pd.read_json (input,lines=True)
    a, b = np.split(df, [int(por*len(df))])
    if lower:
        for index, row in a.iterrows():
                row['text'] =list(map(lambda x: x.lower(), row['text']))
            
    df=a
    count=0
    t=False
    if balance:
                for index,row in df.iterrows():
                    l=row['label']
                    #Balance Premise
                    
                    if 'I-PER'in l and 'I-LOC' not in l and 'I-ORG' not in l and 'I-MISC' not in l  and len(ig)<=901:
                        if index not in ig:
                            ig.append(index)
                    if 'I-LOC' in l and 'I-PER' not in l and 'I-ORG'not in l and 'I-MISC' not in l and len(ik)<=546:
                            ig.append(index)
                    if 'I-ORG' in l and 'I-PER' not in l and 'I-LOC'not in l and 'I-MISC' not in l and len(ih)<=253:
                        if index not in ig:
                            ig.append(index)
                    #Balance O   
                    if count<=1847 and t:
                        count=count+l.count('O')
                        if index not in ig:
                            it.append(index)
                df=df.drop(ig)
                df=df.drop(it)
    df.to_json(output,lines=True,orient='records')

input='/home/jon/Documentos/TFM/work/EntLM-main/dataset/realconll/balance/10/train.json'
output='/home/jon/Documentos/TFM/work/EntLM-main/dataset/realconll/balance/lower/10/train.json'
por=1
lower=True
balance=False
split(input,output,por,lower,balance)


