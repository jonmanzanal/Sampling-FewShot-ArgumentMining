import csv
import pandas as pd
import numpy as np
import sys
import os
import collections
from transformers import BasicTokenizer

def ckeckList(lst):
 
    ele = lst[0]
    chk = True
 
    # Comparing each element with first item
    for item in lst:
        if ele != item:
            chk = False
            break
 
    if (chk == True):
        return True
    else:
        return False
def read_ssv(input_file, output,train,por,balance,lower,replace=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = f.readlines()
            lines.append("\n") #workaround adding a stop criteria for last sentence iteration

            sentences = []
            try:
                lines[0].split(' ')[1]
            except IndexError as err:
                print('Label column', err)
                raise

            tokenizer = BasicTokenizer(do_lower_case=lower)
            sent_tokens = []
            sent_labels = []

            for line in lines:

                line = line.split(' ')

                if len(line) < 2:
                    assert len(sent_tokens) == len(sent_labels)
                    if sent_tokens == []:
                        continue

                    if replace == None:
                        sentences.append([sent_tokens, sent_labels])
                    else:
                        sent_labels = [replace[label] if label in replace.keys() else label for label in sent_labels]
                        sentences.append([' '.join(sent_tokens), sent_labels])
                    sent_tokens = []
                    sent_labels = []
                    continue

                token = line[0]
                label = line[1].replace('\n', '')
                tokenized = tokenizer.tokenize(token)

                if len(tokenized) > 1:

                    for i in range(len(tokenized)):
                        if 'B-' in label:
                            if i < 1:
                                sent_tokens.append(tokenized[i])
                                sent_labels.append(label)
                            else:
                                sent_tokens.append(tokenized[i])
                                #sent_labels.append(label.replace('B-', 'I-')) #if only the first token should be B-
                                sent_labels.append(label)
                        else:
                            sent_tokens.append(tokenized[i])
                            sent_labels.append(label)

                else:
                    sent_tokens.append(tokenized[0])
                    sent_labels.append(label)

        df = pd.DataFrame(sentences,columns = ['text','label'])
        ig=[]
        it=[]
        count=0
        t=True
        if train:
            a, b = np.split(df, [int(por*len(df))])
            df=a
            if balance:
                for index,row in df.iterrows():
                    l=row['label']
                    #Balance Premise
                    #5% 45
                    #10% 87
                    if l[0] == 'B-Premise' and 'B-Claim' not in l and len(ig)<=87:
                        ig.append(index)
                    #Balance O   
                    if ckeckList(l) and count<=2631 and t:
                        count=count+l.count('O')
                        it.append(index)
                df=df.drop(ig)
                df=df.drop(it)
        f = open(output, "w")
        for row in df.iterrows():
            row[1].to_json(f)
            f.write("\n")
        f.close()

input_dir = '/home/jon/Documentos/TFM/ecai2020-transformer_based_am-master/data/argument_components/manual_projections/deepl/awesome/train.tsv'
#output = '/home/jon/Documentos/TFM/EntLM-main/dataset/mine/train.json'
output = '/home/jon/Documentos/TFM/ecai2020-transformer_based_am-master/data/argument_components/manual_projections/deepl/awesome/train.json'
train=False
balance=False
lower=False
por=1
read_ssv(input_dir, output,train,por,balance,lower)


