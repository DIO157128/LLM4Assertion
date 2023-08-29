import os

import pandas as pd


def compare():
    df1 = pd.read_csv('../fine_tune_data/assert_test.csv')
    target = df1['target'].tolist()
    filter = []
    for t in target:
        if t.startswith('Assert') or t.startswith('assert'):
            filter.append(1)
        else:
            filter.append(0)
    f = open('CodeT5/CodeT5_beam50.txt', 'r', encoding='utf-8')
    preds = f.read().splitlines()
    match = []
    for i in range(len(preds)):
        if preds[i]=="match:":
            match.append(int(preds[i+1]))
    filter_match = []
    assert len(filter)==len(match)
    for f,m in zip(filter,match):
        if f:
            filter_match.append(m)
    assert len(filter_match)!=len(match)
    print(sum(filter_match)/len(filter_match))
    print(len(filter_match))
    print(len(match))
def filterori():
    df1 = pd.read_csv('../fine_tune_data/assert_test.csv')
    target = df1['target'].tolist()
    print(len(target))
    filter = []
    for t in target:
        if t.startswith('Assert') or t.startswith('assert'):
            filter.append(1)
        else:
            filter.append(0)
    f = open('CodeT5/CodeT5_beam1.txt', 'r', encoding='utf-8')
    preds = f.read().splitlines()
    f.close()
    match = []
    for i in range(len(preds)):
        if preds[i] == "match:":
            match.append(int(preds[i + 1]))
    filter_match = []
    assert len(filter) == len(match)
    for f, m in zip(filter, match):
        if f:
            filter_match.append(m)
    assert len(filter_match) != len(match)
    f = open('CodeT5/CodeT5_mask_beam1.txt','r',encoding='utf-8')
    preds_mask = f.read().splitlines()
    f.close()
    mask_match = []
    for i in range(len(preds_mask)):
        if preds_mask[i] == "match:":
            mask_match.append(int(preds_mask[i + 1]))
    assert len(mask_match)==len(filter_match)
    diff_index = []
    i = 0
    for mm,fm in zip(mask_match,filter_match):
        if mm ==0 and fm==1:
            diff_index.append(i)
        i+=1
    print(diff_index)
def find_string_in_list(strings, target_string):
    for index, string in enumerate(strings):
        if target_string in string:
            return index
    return -1
def getexample():
    df1 = pd.read_csv('../fine_tune_data/assert_test.csv')
    target = df1['target'].tolist()
    print(len(target))
    filter = []
    for t in target:
        if t.startswith('Assert') or t.startswith('assert'):
            filter.append(1)
        else:
            filter.append(0)
    f = open('CodeT5/CodeT5_beam1.txt', 'r', encoding='utf-8')
    preds = f.read().splitlines()
    f.close()
    match = []
    for i in range(len(preds)):
        if preds[i] == "match:":
            match.append(int(preds[i + 1]))
    filter_match = []
    assert len(filter) == len(match)
    for f, m in zip(filter, match):
        if f:
            filter_match.append(m)
    assert len(filter_match) != len(match)
    f = open('CodeT5/CodeT5_mask_beam1.txt','r',encoding='utf-8')
    preds_mask = f.read().splitlines()
    f.close()
    mask_match = []
    for i in range(len(preds_mask)):
        if preds_mask[i] == "match:":
            mask_match.append(int(preds_mask[i + 1]))
    assert len(mask_match)==len(filter_match)
    diff_index = []
    i = 0
    for mm,fm in zip(mask_match,filter_match):
        if mm ==0 and fm==0:
            diff_index.append(i)
        i+=1
    print(diff_index)
    print(len(diff_index))
    f1 = open('CodeT5/CodeT5_beam1.txt','r',encoding='utf-8')
    f2 = open('CodeT5/CodeT5_mask_beam1.txt','r',encoding='utf-8')
    finetune_lines = f1.read().splitlines()
    mask_lines = f2.read().splitlines()
    df = pd.read_csv('../fine_tune_data/assert_mask_new_test.csv')
    target = df['target']
    f3 = open('mask_f_finetune_f.txt','a',encoding='utf-8')
    for i in diff_index[:100]:
        t = target[i]
        f_index = find_string_in_list(finetune_lines,t)
        m_index = find_string_in_list(mask_lines,t)
        i_result = []
        i_result.append('{}:'.format(i))
        i_result.append('mask:')
        f_lines = finetune_lines[f_index-3:f_index+5]
        m_lines = mask_lines[m_index - 3:m_index + 5]
        i_result+=m_lines
        i_result.append('finetune:')
        i_result+=f_lines
        i_result.append('-'*20)
        for item in i_result:
            f3.write(str(item) + '\n')
if __name__ == '__main__':
    getexample()