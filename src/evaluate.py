from metric import grammar_correctness, readability_score
from utils import load_txt
from metric import rouge, meteor, bert_score, s_bert, BLEURT
from statistic import load_fields
from tqdm import tqdm
import os
import numpy as np


def evaluate_gc_read(p):
    file_list = os.listdir(p)
    for file in file_list:
        print('{}'.format(file[:-4]), end=',')
        swd = load_txt(os.path.join(p, file))
        gc = grammar_correctness(swd)
        read = readability_score(swd)
        print('{},{}'.format(gc, read))


def evaluate_similarity(swd_d, ground_truth_dir, filed_p):
    fs = os.listdir(swd_d)
    field_m = load_fields(filed_p)
    m = {
        'Industry': [],
        'Finance': [],
        'Logistics': [],
        'Pharmacy': [],
        'Insurance': [],
        'Education': [],
        'Others': []
    }
    for f in tqdm(fs):
        swd = load_txt(os.path.join(swd_d, f))
        gt = ''
        for gt_d in ground_truth_dir:
            if os.path.exists(os.path.join(gt_d, f)):
                gt = load_txt(os.path.join(gt_d, f))
        rg = rouge(swd, gt)
        me = meteor(swd, gt)
        # bs = bert_score(swd, gt)
        # bl = BLEURT(swd, gt)
        # sb = s_bert(swd, gt)
        # scores = [bs, sb, bl]
        scores = [rg, me]
        m[field_m.get(f[:-3] + 'xml')].append(scores)
    for k, v in m.items():
        print(k)
        v = np.array(v)
        avg = np.average(v, axis=0)
        for s in avg:
            print(round(s, 2))


if __name__ == '__main__':
    evaluate_similarity('result/HCT', ['dataset/n_swd', 'dataset/swd'], 'result/fields_96.txt')
