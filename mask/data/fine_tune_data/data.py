import csv
import re

import pandas as pd
import torch

def extract_java_method_name(input_str):
    # 匹配类似于 assertEquals 这样的方法名，可能包含空格
    return input_str[0:input_str.find('(')]
def extract_java_method_arguments(input_str):
    return input_str[input_str.find('(')+1:input_str.rfind(')')]
def transTxt4mask(name,path1,path2):
    f1 = open(path1,'r',encoding='utf-8')
    f2 =  open(path2,'r',encoding='utf-8')
    source = f1.readlines()
    target = f2.readlines()
    source = [s.strip() for s in source]
    target = [t.strip() for t in target]
    processed_s = []
    processed_t = []
    for s,t in zip(source,target):
        if t.startswith('Assert') or t.startswith('assert'):
            test_prefix = s.split('"<FocalMethod>"')[0].strip()
            focal_method = s.split('"<FocalMethod>"')[1].strip()
            asserttype = extract_java_method_name(t)
            processed_s.append('"<FocalMethod>" '+focal_method + '"<TestPrefix>" '+test_prefix.replace('"<AssertPlaceHolder>"',asserttype + " ( <mask> ) "))
            processed_t.append(extract_java_method_arguments(t))
    df = pd.DataFrame()
    df['source'] = processed_s
    df['target'] = processed_t
    df.to_csv("assert_mask_new_{}.csv".format(name),index=False,escapechar="\\",encoding='utf-8')
if __name__ == '__main__':
    # transTxt4mask("train","./NewDataSet/Training/testMethods.txt","./NewDataSet/Training/assertLines.txt")
    # transTxt4mask("val", "./NewDataSet/Validation/testMethods.txt", "./NewDataSet/Validation/assertLines.txt")
    # transTxt4mask("test", "./NewDataSet/Testing/testMethods.txt", "./NewDataSet/Testing/assertLines.txt")
    # df = pd.read_csv('assert_mask_new_train.csv')
    df1 = pd.read_csv('assert_train.csv')
    df2 = pd.read_csv('assert_val.csv')
    df3 = pd.read_csv('assert_test.csv')
    print('train :{}'.format(len(df1)))
    print('val :{}'.format(len(df2)))
    print('test :{}'.format(len(df3)))