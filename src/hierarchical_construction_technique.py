from utils import load_xml, load_txt, save_as_txt
from call_llm import prompt_llm
import time
import xml.etree.ElementTree as ET
import os
import metric


def rigid(node, swds, instruction, output_indicator):
    rigid_ = load_txt('prompt/hct/rigid/rigid.txt')
    p = '{}{}{}# Input\n<?xml version="1.0" encoding="UTF-8"?>\n<rpst>\n\t<seq>\n\t\t{}</seq>\n</rpst>'.format(instruction, rigid_,output_indicator, ET.tostring(node, encoding='utf-8', method='xml').decode('utf-8')[:-4])
    # p = '{}{}# Input\n{}'.format(instruction, output_indicator, ET.tostring(node, encoding='utf-8', method='xml').decode('utf-8'))
    return p


def xor(node, swds, instruction, output_indicator):
    p = ''
    if node.attrib['type'] == 'skip':
        xor_skip = load_txt('prompt/hct/xor/xor_skip.txt')
        condition = node.attrib['condition'].split('|')[1:-1]
        description = node.attrib['description']
        p = '{}{}{}# Input\n'.format(instruction, xor_skip, output_indicator)
        if description == '':
            p += 'if {}, """{}""" if {}, skip.'.format(condition[1], swds[0][0], condition[0])
        else:
            p += '{} if {}, """{}""" if {}, skip.'.format(description, condition[1], swds[0][0], condition[0])
    elif node.attrib['type'] == '' or node.attrib['type'] == 'event based':
        xor_normal_event_based = load_txt('prompt/hct/xor/xor_normal_event_based.txt')
        xor_empty = load_txt('prompt/hct/xor/xor_empty.txt')
        condition = node.attrib['condition'].split('|')[1:-1]
        description = node.attrib['description']
        empty = True
        for c in condition:
            if c != '':
                empty = False
                break
        if empty:
            p = '{}{}{}# Input\n'.format(instruction, xor_empty, output_indicator)
            for i in range(len(condition)):
                p += '"""{}""" xor '.format(swds[i][0])
            p = p[:-5]
        else:
            p = '{}{}{}# Input\n'.format(instruction, xor_normal_event_based, output_indicator)
            if description == '':
                for i in range(len(condition)):
                    p += 'if {}, """{}""" '.format(condition[i], swds[i][0])
                p = p[:-1]
            else:
                p += '{} '.format(description)
                for i in range(len(condition)):
                    p += 'if {}, """{}""" '.format(condition[i], swds[i][0])
                p = p[:-1]
    return p


def seq(node, swds, instruction, output_indicator):
    flag = False
    for i in range(len(swds)):
        if (swds[i][1] == 'xor' or swds[i][1] == 'and') and i != len(swds) - 1:
            flag = True
    p = ''
    if flag:
        sequence_exclusive_parallel_end = load_txt('prompt/hct/seq/sequence_exclusive_parallel_end.txt')
        p = '{}{}{}# Input\n'.format(instruction, sequence_exclusive_parallel_end, output_indicator)
        for swd in swds:
            if swd[1] == 'xor':
                p += '"""{}""" exclusive choice end, next '.format(swd[0])
            elif swd[1] == 'and':
                p += '"""{}""" parallel end, next '.format(swd[0])
            else:
                p += '"""{}""" next '.format(swd[0])
        p = p[:-6]
    else:
        sequence = load_txt('prompt/hct/seq/sequence.txt')
        p = '{}{}{}# Input\n'.format(instruction, sequence, output_indicator)
        for swd in swds:
            p += '"""{}""" next '.format(swd[0])
        p = p[:-6]
    return p


def and_(node, swds, instruction, output_indicator):
    and_ = load_txt('prompt/hct/and/and.txt')
    p = '{}{}{}# Input\n'.format(instruction, and_, output_indicator)
    for swd in swds:
        p += '"""{}""" and '.format(swd[0])
    p = p[:-5]
    return p


def loop(node, swds, instruction, output_indicator):
    p = ''
    if node.attrib['type'] == 'while':
        while_ = load_txt('prompt/hct/loop/dowhile.txt')
        description = node.attrib['description']
        condition = node.attrib['condition']
        exit_ = node.attrib['exit']
        p = '{}{}{}# Input\n'.format(instruction, while_, output_indicator)
        if description == '':
            p += '{} if {}, loop """{}""" if {}, exit.'.format(description, condition, swds[0][0], exit_)
        else:
            p += 'if {}, loop """{}""" if {}, exit.'.format(condition, swds[0][0], exit_)
    else:
        description = node.attrib['description']
        condition = node.attrib['condition']
        exit_ = node.attrib['exit']
        if len(swds) > 1:
            dowhile_post = load_txt('prompt/hct/loop/dowhile_post.txt')
            p = '{}{}{}# Input\n'.format(instruction, dowhile_post, output_indicator)
            if description == '':
                p += '"""{}""" if {}, """{}""" loop, if {}, exit.'.format(swds[0][0], condition, swds[1][0], exit_)
            else:
                p += '"""{}""" {} if {}, """{}""" loop. if {}, exit.'.format(swds[0][0], description, condition,
                                                                             swds[1][0], exit_)
        else:
            dowhile = load_txt('prompt/hct/loop/dowhile.txt')
            p = '{}{}{}# Input\n'.format(instruction, dowhile, output_indicator)
            if description == '':
                p += '"""{}""" if {}, loop. if {}, exit.'.format(swds[0][0], condition, exit_)
            else:
                p += '"""{}""" {} if {}, loop. if {}, exit.'.format(swds[0][0], description, condition, exit_)
    return p


def or_(node, swds, instruction, output_indicator):
    p = ''
    condition = node.attrib['condition'].split('|')[1:-1]
    empty = True
    for c in condition:
        if c != '':
            empty = False
            break
    if empty:
        or_empty = load_txt('prompt/hct/or/or_empty.txt')
        p = '{}{}{}# Input\n'.format(instruction, or_empty, output_indicator)
        for swd in swds:
            p += '"""{}""" or '.format(swd)
        p = p[:-4]
    else:
        or_ = load_txt('prompt/hct/or/or.txt')
        p = '{}{}{}# Input\n'.format(instruction, or_, output_indicator)
        for i in range(len(condition)):
            p += 'if {}, """{}""" '.format(condition[i], swds[i][0])
        p = p[:-1]
    return p


def gen_swd(node, sub_swds):
    model = 'gpt-3.5-turbo'
    pre = {'role': 'system', 'content': 'You are an assistant who follows instructions.'}
    temperature = 0
    instruction_path = 'prompt/hct/instruction.txt'
    output_indicator_path = 'prompt/hct/output_indicator.txt'
    i = load_txt(instruction_path)
    i_rigid = load_txt('prompt/hct/instruction_rigid.txt')
    o = load_txt(output_indicator_path)

    for swd in sub_swds:
        swd[0] = swd[0].strip()

    p = None
    if node.tag == 'rigid':
        p = rigid(node, sub_swds, i_rigid, o)
    elif node.tag == 'xor':
        p = xor(node, sub_swds, i, o)
    elif node.tag == 'seq':
        if len(sub_swds) == 1:
            return sub_swds[0][0]
        p = seq(node, sub_swds, i, o)
    elif node.tag == 'and':
        p = and_(node, sub_swds, i, o)
    elif node.tag == 'loop':
        p = loop(node, sub_swds, i, o)
    elif node.tag == 'or':
        p = or_(node, sub_swds, i, o)
    while (True):
        try:
            swd = prompt_llm(model, pre, p, temperature)
            return swd
        except Exception:
            print('prompt problem')
            print(Exception)
            time.sleep(1)
            continue


def hct(root):
    if root.tag == 'task':
        if root.attrib['lane'] == '' and root.attrib['pool'] != '' and root.attrib['pool'] != '?':
            return '{} {}'.format(root.attrib['pool'], root.text)
        elif root.attrib['lane'] != '' and root.attrib['pool'] != '' and root.attrib['pool'] != '?':
            return '{} of {} {}'.format(root.attrib['lane'], root.attrib['pool'], root.text)
        elif root.attrib['lane'] == '' and (root.attrib['pool'] == '' or root.attrib['pool'] == '?'):
            return '{}'.format(root.text)
    elif root.tag == 'rigid':
        return gen_swd(root, '')
    else:
        sub_swds = []
        for child in root:
            sub_swd = hct(child)
            sub_swds.append([sub_swd, child.tag])
        return gen_swd(root, sub_swds)


def print_tree(root):
    print(root)
    for child in root:
        print_tree(child)


def main_process():
    rpst_dir = 'dataset/n_rpst'
    ground_truth_dir = 'dataset/n_swd'
    save_dir = 'result/HCT'

    # test
    files = ['dataset/n_rpst/3217_1e1304b008124ea5a8c60ba8c3c09e9b.xml',
             'dataset/n_rpst/42_40debacaaa344ff4adb39cdfdb748f57.xml',
             'dataset/n_rpst/250_4103bf5e18db4e89abb442d9b279f005.xml',
             'dataset/tested/848_1c49ec4b4ea8468e8b553c95564f8b4c --created--.xml','dataset/tested/911_1c55cc7cfebb47febb38d36f6423adeb --created--.xml','dataset/tested/2680_1daef3f1695441929acacc6eb4e278ad --created--.xml','dataset/tested/2699_1db17a9f65344bc99afc3ef3e5df2689 --created--.xml',
             'dataset/tested/Refined Fulfil Prescription --created--.xml']
    for f in files:
        rpst = load_xml(f)
        for root in rpst:
            swd = hct(root)
            print()

    i = 0
    file_list = os.listdir(rpst_dir)
    for file in range(len(file_list)):
        rpst = load_xml(os.path.join(rpst_dir, file_list[file + i]))
        print('{}'.format(file_list[file + i][:-4]), end=',')
        # ground_truth = load_txt(os.path.join(ground_truth_dir, file[:-3] + '.txt'))
        for root in rpst:
            swd = hct(root)
            # grammar correctness
            gc = metric.grammar_correctness(swd)
            # readability
            read = metric.readability_score(swd)
            # meteor
            # mt = metric.meteor(swd, ground_truth)
            # rouge
            # rg = metric.rouge(swd, ground_truth)
            # bert score
            # bs = metric.bert_score(swd, ground_truth)
            # sentence bert
            # sbert = metric.s_bert(swd, ground_truth)
            # save_as_txt(os.path.join(save_dir, file.replace('xml', 'txt')), swd)
            # print('rpst: {file}, grammar correctness: {gc}, readability: {read}, meteor: {mt}, rouge: {rg}, bert score: {bs}, sentence bert: {sbert}'.format(file=file, gc=gc, read=read, mt=mt, rg=rg, bs=bs, sbert=sbert))
            print('{},{}'.format(gc,read))
            save_as_txt(os.path.join(save_dir, file_list[file + i].replace('xml', 'txt')), swd)


if __name__ == '__main__':
    main_process()
