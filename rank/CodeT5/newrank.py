import pandas as pd
df = pd.read_csv('assertion_preds.csv')
grouped = [df[i:i+10] for i in range(0, len(df), 10)]
match = []
pred = []
for group in grouped:
    group.sort_values(by='logit_1', ascending=False, inplace=True)
    pred.append(group.iloc[0]['pred_assertion'])
df2 = pd.read_csv('../data/fine_tune_data/assert_test.csv')
df2['pred'] = pred
target = df2['target']
match = [p.replace(" ", "")==t.replace(" ", "") for p,t in zip(pred,target)]
df2['match'] = match
df2.to_csv('CodeT5_pred.csv')
print(sum(match)/len(match))