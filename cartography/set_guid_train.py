from datasets import load_dataset
import pandas as pd

df = pd.read_table('/Users/joey/Downloads/snli_1.0/train.tsv')
# print(len(df))
df = df[['#' in pID for pID in list(df['pairID'])]]

df.index = range(len(df))

for idx in range(len(df)):

    string = df.at[idx, 'pairID']
    ls = string.split('#')
    # print(ls)
    after = ls[1]
    before = ls[0]
    period_split = before.split('.')
    before = period_split[0]
    after = list(after)
    if after[3] == 'e':
        after[3] = '0'
    elif after[3] == 'n':
        after[3] = '1'
    else:
        after[3] = '2'

    after = after[0] + after[2] + after[3]
    df.at[idx, 'pairID'] = before + after
print(df['pairID'])
df.to_csv('train-guid.tsv',sep='\t')
# dataset = load_dataset("glue", 'mnli')
# print(dataset['train']['idx'])