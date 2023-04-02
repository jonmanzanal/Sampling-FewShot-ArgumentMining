import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def split(input,output,por,lower,balance):
    ig=[]
    it=[]
    df = pd.read_csv(input, sep='\t', encoding='utf-8',names=['Label','text','text1'])
    a, b = np.split(df, [int(por*len(df))])
    if lower:
        for index, row in a.iterrows():
                row['text'] =list(map(lambda x: x.lower(), row['text']))
            
    df=a
    df1 = df['Label'].value_counts()
    print('Before',por)
    print('Before',df1)
    t=False
    g=True
    if balance:
                for index,row in df.iterrows():
                    l=row['Label']
                    #Balance Premise
                    
                    if '__label__Support' in l and len(ig)<=87 and t:
                        if index not in ig:
                            ig.append(index)

                    #Balance Rel   
                    if '__label__noRel' in l and len(it)<=516 and g:
                        if index not in ig:
                            it.append(index)
                df=df.drop(ig)
                df=df.drop(it)
    df1 = df['Label'].value_counts()
    print('After',por)
    print('After',df1)
    df.to_csv(output,sep='\t', encoding='utf-8', index=False, header=False)

input='/home/jon/Documentos/TFM/ecai2020-transformer_based_am-master/preprocessing/rel/original/train_relations.tsv'
output='/home/jon/Documentos/TFM/ecai2020-transformer_based_am-master/preprocessing/rel/balance/unlabeled_relations.tsv'
por=0.1
lower=False
balance=True
split(input,output,por,lower,balance)




