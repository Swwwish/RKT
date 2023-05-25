import pandas as pd
import numpy as np
import itertools
import os

def sta_infos(df, keys, stares, split_str="_"):
    # keys: 0: uid , 1: concept, 2: question
    uids = df[keys[0]].unique()
    if len(keys) == 2:
        cids = df[keys[1]].unique()
    elif len(keys) > 2:
        qids = df[keys[2]].unique()
        ctotal = 0
        cq = df.drop_duplicates([keys[2], keys[1]])[[keys[2], keys[1]]]
        cq[keys[1]] = cq[keys[1]].fillna("NANA")
        cids, dq2c = set(), dict()
        for i, row in cq.iterrows():
            q = row[keys[2]]
            ks = row[keys[1]]
            dq2c.setdefault(q, set())
            if ks == "NANA":
                continue
            for k in str(ks).split(split_str):
                dq2c[q].add(k)
                cids.add(k)
        ctotal, na, qtotal = 0, 0, 0
        for q in dq2c:
            if len(dq2c[q]) == 0:
                na += 1 # questions has no concept
                continue
            qtotal += 1
            ctotal += len(dq2c[q])
        
        avgcq = round(ctotal / qtotal, 4)
    avgins = round(df.shape[0] / len(uids), 4)
    ins, us, qs, cs = df.shape[0], len(uids), "NA", len(cids)
    avgcqf, naf = "NA", "NA"
    if len(keys) > 2:
        qs, avgcqf, naf = len(qids), avgcq, na
    curr = [ins, us, qs, cs, avgins, avgcqf, naf]
    stares.append(",".join([str(s) for s in curr]))
    return ins, us, qs, cs, avgins, avgcqf, naf

def write_txt(file, data):
    with open(file, "w") as f:
        for dd in data:
            for d in dd:
                f.write(",".join(d) + "\n")

from datetime import datetime
def change2timestamp(t, hasf=True):
    if hasf:
        timeStamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000
    else:
        timeStamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timestamp() * 1000
    return int(timeStamp)

def replace_text(text):
    text = text.replace("_", "####").replace(",", "@@@@")
    return text


def format_list2str(input_list):
    return [str(x) for x in input_list]


def one_row_concept_to_question(row):
    """Convert one row from concept to question

    Args:
        row (_type_): _description_

    Returns:
        _type_: _description_
    """
    new_question = []
    new_concept = []
    new_response = []

    tmp_concept = []
    begin = True
    for q, c, r, mask, is_repeat in zip(row['questions'].split(","),
                                        row['concepts'].split(","),
                                        row['responses'].split(","),
                                        row['selectmasks'].split(","),
                                        row['is_repeat'].split(","),
                                        ):
        if begin:
            is_repeat = "0"
            begin = False
        if mask == '-1':
            break
        if is_repeat == "0":
            if len(tmp_concept) != 0:
                new_concept.append("_".join(tmp_concept))
                tmp_concept = []
            new_question.append(q)
            new_response.append(r)
            tmp_concept = [c]
        else:#如果是 1 就累计知识点
            tmp_concept.append(c)
    if len(tmp_concept) != 0:
        new_concept.append("_".join(tmp_concept))

    if len(new_question) < 200:
        pads = ['-1'] * (200 - len(new_question))
        new_question += pads
        new_concept += pads
        new_response += pads

    new_selectmask = ['1']*len(new_question)
    new_is_repeat = ['0']*len(new_question)

    new_row = {"fold": row['fold'],
               "uid": row['uid'],
               "questions": ','.join(new_question),
               "concepts": ','.join(new_concept),
               "responses": ','.join(new_response),
               "selectmasks": ','.join(new_selectmask),
               "is_repeat": ','.join(new_is_repeat),
               }
    return new_row

def concept_to_question(df):
    """Convert df from concept to question
    Args:
        df (_type_): df contains concept

    Returns:
        _type_: df contains question
    """
    new_row_list = list(df.apply(one_row_concept_to_question,axis=1).values)
    df_new = pd.DataFrame(new_row_list)
    return df_new

def get_df_from_row(row):
    value_dict = {}
    for col in ['questions', 'concepts', 'responses', 'is_repeat']:
        value_dict[col] = row[col].split(",")
    df_value = pd.DataFrame(value_dict)
    df_value = df_value[df_value['questions']!='-1']
    return df_value


# pykt version (new)
# phi coefficient
def cal_table(table_dict, concept, correct, i, j):
    if i not in table_dict:
        table_dict[i] = {}
        
    i_index_list = np.where(concept == i)[0]  # np.where()返回一个包含知识点i下标的元组
    for i_idx in i_index_list:
        temp_c = concept[:i_idx]   
        if j in temp_c:
            if j not in table_dict[i]:
                table_dict[i][j] = np.zeros([2,2], dtype=float)
            table = table_dict[i][j]
            j_index = np.where(temp_c == j)[0][-1]   # 统计在知识点i之前“最近”的知识点j
            table[ correct[i_idx], correct[j_index] ] += 1.0
            table_dict[i][j] = table
        

def compute_phi_corr(df, dpath):
    keys = df.columns
    if 'timestamps' not in keys:
        print("the dataset doesn't have the time information")
        return
    
    for index in keys:
        if index in ["fold", "uid"]:
            continue
        df[index] = df[index].apply(lambda x: x.split(','))
    
    if 'timestamps' and 'questions' in keys:
        df_ = df.explode(['questions','concepts','responses','timestamps'])
    elif 'questions' not in keys:
        df_ = df.explode(['concepts','responses','timestamps'])
    else:
        df_ = df.explode(['questions','concepts','responses'])
    print(df_)    
    
    # phi calculate
    phi_dict = dict()
    table_dict = dict()
    count=0
    
    for stu,stu_df in df_.groupby(['uid'], sort=False):
        #stu_df.sort_values(by=['timestamps'], ascending=True , inplace=True)
        count+=1
        
        question = np.array( [int(q) for q in stu_df['questions'].values] )
        qu = np.array( [int(q) for q in stu_df['questions'].unique()] )
        #print(f"concept:{concept}")
        correct = [int(s) for s in stu_df['responses'].values]
        #print(f"response:{correct}")
        
        print(f"No.{count} :stu_id:{stu} , question num:{len(stu_df)}")
        combinations = list(itertools.product(qu, qu))
        
        # 循环遍历所有知识点i j的组合
        for i,j in combinations:
            cal_table(table_dict, question, correct, i, j)
            
    pd.to_pickle( table_dict, os.path.join(dpath, "table_dict.pkl") )
    
    # 计算全局phi    
    for i in table_dict.keys():
        if len(table_dict[i]) == 0:
                continue
        for j in table_dict[i].keys():
            table = table_dict[i][j]
            row_sum = table.sum(axis=1)
            culumn_sum = table.sum(axis=0)

            if 0.0 in row_sum or 0.0 in culumn_sum:
                continue
            else:
                mul = np.prod(row_sum) * np.prod(culumn_sum)
                phi = (table[0][0]*table[1][1] - table[0][1]*table[1][0]) / np.sqrt( mul )
                if phi == 0.0:
                    continue
                if i not in phi_dict:
                    phi_dict[i] = {}
                phi_dict[i][j] = phi
                
    pd.to_pickle( phi_dict, os.path.join(dpath, "phi_dict.pkl") )        


    