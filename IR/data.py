import pandas as pd
import json


def getIRdata(type):
    # 打开 JSONL 文件
    searched_assertion = []
    with open('NewDataset/{}.jsonl'.format(type), 'r', encoding='utf-8') as file:
        for line in file:
            # 解析每一行的 JSON 对象
            data = json.loads(line)
            # 现在可以使用 data 变量来访问每个 JSON 对象的内容
            searched_assertion.append(data['src_desc'])
    df = pd.read_csv('assert_{}_new.csv'.format(type))
    source = df['source']
    add_source = []
    assert len(source) == len(searched_assertion)
    for s, ss in zip(source,searched_assertion):
        search_string = '"<AssertPlaceHolder>" ;'
        index = s.find(search_string)
        modified_string = s[:index + len(search_string)] + ' /* ' + ss + ' */ '+ s[
                                                                       index + len(
                                                                           search_string):]

        add_source.append(modified_string)
    df['source'] = add_source
    df.to_csv('assert_{}_new_IR.csv'.format(type),index=False)
if __name__ == '__main__':
    getIRdata('test')
    getIRdata('val')
    getIRdata('train')
