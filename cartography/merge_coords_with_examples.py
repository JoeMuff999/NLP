import pandas as pd

coords_df = pd.read_json('data/data_map_coordinates/snli_roberta_0_6_data_map_coordinates.jsonl', lines=True)
train_df = pd.read_table('train-guid.tsv')
# print(train_df.columns)
train_df = train_df.drop('Unnamed: 0', axis=1)
train_df = train_df.rename(columns={"sentence1":"premise", "sentence2":"hypothesis", "gold_label" : "label"}) # rename to match huggingface snli

train_df['index'] = ''
train_df['confidence'] = ''
train_df['variability'] = ''


num_not_found = 0

for idx in range(len(train_df)):
    
    rowIdx = coords_df.index[coords_df['guid'] == train_df.at[idx, 'pairID']].tolist()
    if len(rowIdx) != 0:
        rowIdx = rowIdx[0]
    elif len(rowIdx) > 1:
        print("this should not happen")
    elif len(rowIdx) == 0:
        num_not_found += 1
    row = coords_df.iloc[rowIdx]
    # print(row['index'])
    # print(row['confidence'])
    train_df.at[idx, 'index'] = row['index'].astype('int64')
    train_df.at[idx, 'confidence'] = row['confidence']
    train_df.at[idx, 'variability'] = row['variability']
    # print(train_df)
    # break
    # print(train_df)


train_df.to_csv('test.csv')
print(train_df)
print('num not found ' + str(num_not_found)) # 791 - this is the number we expected 