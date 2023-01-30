import random
import csv
import pandas as pd
def sample_data(data_path, output_path, k=10):
    with open(data_path, 'r') as f:
        few_shot_data = []
        df = pd.read_csv(data_path,sep='\t',names=['Label','text','text1'])
        df = df.sample(frac = 1)
        sup=[]
        att=[]
        rel=[]
        for index,row in df.iterrows():
            if row['Label'] == '__label__Support' and len(sup) < k:
                sup.append(row)
                few_shot_data.append(row)
            if row['Label'] == '__label__Attack' and len(att) < k:
                att.append(row)
                few_shot_data.append(row)
            if row['Label'] == '__label__noRel' and len(rel) < k:
                rel.append(row)
                few_shot_data.append(row)

    with open(output_path, 'w', newline='',encoding='utf-8') as wf:
        writer = csv.writer(wf,delimiter='\t',lineterminator="\n")
        for row in few_shot_data:
            writer.writerow(row)
            




if __name__ == '__main__':
    import os
    print("Running")
    for k in [50]:
        for i in range(1,4):
            path = f"/home/jon/Documentos/TFM/ecai2020-transformer_based_am-master/preprocessing/rel/{k}shot/{i}"
            if not os.path.exists(path):
                print("Create")
                os.makedirs(path)

            sample_data(f"/home/jon/Documentos/TFM/ecai2020-transformer_based_am-master/preprocessing/rel/original/header/train_relations.tsv", f"{path}/train_relations.tsv", k=k)
            print("Done")