from utils import load_xml, load_txt
from call_llm import prompt_llm
from matplotlib import pyplot as plt
from metric import readability_score
import os
import numpy


def round_2(number):
    return round(number, 2)


def dfs(root, m):
    if root.tag == 'rigid':
        m['rigid'] += 1
    if root.tag == 'seq':
        m['seq'] += 1
    if root.tag == 'xor':
        m['xor'] += 1
    if root.tag == 'and':
        m['and'] += 1
    if root.tag == 'or':
        m['or'] += 1
    if root.tag == 'loop':
        m['loop'] += 1
    if root.tag == 'task' or root.tag == 'event':
        m['task'] += 1
    for child in root:
        dfs(child, m)


def nodes_num(p, m):
    file_list = os.listdir(p)
    for f in file_list:
        rpst = load_xml(os.path.join(p, f))
        for root in rpst:
            dfs(root, m)
    return m


def multi_dir_nodes_num():
    m = {
        'seq': 0,
        'xor': 0,
        'rigid': 0,
        'and': 0,
        'or': 0,
        'loop': 0,
        'task': 0
    }
    m = nodes_num('dataset/n_rpst', m)
    m = nodes_num('dataset/tested', m)
    m = nodes_num('dataset/ppp', m)
    print(m)


def fields(p, m):
    pre = {'role': 'system', 'content': ''}
    temperature = 0
    file_list = os.listdir(p)
    for f in file_list:
        xml = load_txt(os.path.join(p, f))
        pmt = 'Which field does the following process come from (Industry, Academic, Finance, Government, Logistics, Insurance, Healthcare)? Only output the domain name.\n{}'.format(
            xml)
        r = prompt_llm('gpt-3.5-turbo', pre, pmt, temperature)
        if m.get(r) != None:
            m[r] += 1
        else:
            m[r] = 0
    return m


def fields_(p):
    pre = {'role': 'system', 'content': ''}
    temperature = 0
    file_list = os.listdir(p)
    for f in file_list:
        xml = load_txt(os.path.join(p, f))
        pmt = 'Which field does the following process come from (Industry, Logistics, Education, Insurance, Finance, Healthcare, Pharmacy or Others)? Only Output the field name.\n{}'.format(
            xml)
        r = prompt_llm('gpt-4o', pre, pmt, temperature)
        print(f + ',' + r)


def multi_dir_fields():
    m = {}
    m = fields('dataset/n_rpst', m)
    m = fields('dataset/tested', m)
    m = fields('dataset/ppp', m)
    print(m)


def multi_dir_fields_():
    fields_('dataset/n_rpst')
    fields_('dataset/tested')
    fields_('dataset/ppp')


def fields__(p):
    m = {}
    with open(p, mode='r', encoding='utf-8') as f:
        for l in f.readlines():
            s = l.split(',')
            if m.get(s[1][:-1]) != None:
                m[s[1][:-1]] += 1
            else:
                m[s[1][:-1]] = 1
        print(m)
        f.close()
    return m


def get_max_depth(root, depth=1, max=0):
    if root.tag != 'task' and root.tag != 'event':
        for child in root:
            max = get_max_depth(child, depth + 1, max)
        return max
    else:
        if max < depth:
            max = depth
        return max


def load_fields(field_path):
    m = {}
    with open(field_path, mode='r', encoding='utf-8') as f:
        for l in f.readlines():
            s = l.split(',')
            m[s[0]] = s[1][:-1]
        f.close()
    return m


def statistic_by_field(field_path, p, result_m):
    field_m = load_fields(field_path)
    file_list = os.listdir(p)
    for f in file_list:
        rpst = load_xml(os.path.join(p, f))
        for root in rpst:
            max_depth = get_max_depth(root)
            m = {
                'seq': 0,
                'xor': 0,
                'rigid': 0,
                'and': 0,
                'or': 0,
                'loop': 0,
                'task': 0
            }
            dfs(root, m)
            results = [max_depth, m['seq'], m['xor'], m['and'], m['or'], m['loop'], m['rigid'], m['task']]
            if result_m.get(field_m[f]) == None:
                result_m[field_m[f]] = [results]
            else:
                result_m[field_m[f]].append(results)
    return result_m


def multi_dir_statistic():
    result_m = {}
    result_m = statistic_by_field('result/fields.txt', 'dataset/tested', result_m)
    result_m = statistic_by_field('result/fields.txt', 'dataset/n_rpst', result_m)
    result_m = statistic_by_field('result/fields.txt', 'dataset/ppp', result_m)
    for k, v in result_m.items():
        v = numpy.array(v)
        print(k)
        avg = numpy.round(v.mean(axis=0), 2)
        max_ = numpy.round(numpy.max(v, axis=0), 2)
        min_ = numpy.round(numpy.min(v, axis=0), 2)
        for i in range(len(avg)):
            print(max_[i], avg[i], min_[i])


def task_number_statistic(p, task_num_m):
    file_list = os.listdir(p)
    for f in file_list:
        rpst = load_xml(os.path.join(p, f))
        for root in rpst:
            m = {
                'seq': 0,
                'xor': 0,
                'rigid': 0,
                'and': 0,
                'or': 0,
                'loop': 0,
                'task': 0
            }
            dfs(root, m)
            key = 0
            for k, v in m.items():
                key += v
            if task_num_m.get(key) == None:
                task_num_m[key] = 1
            else:
                task_num_m[key] += 1
    return task_num_m


def multi_dir_task_number_statistic():
    task_num_m = {}
    task_num_m = task_number_statistic('dataset/n_rpst', task_num_m)
    task_num_m = task_number_statistic('dataset/tested', task_num_m)
    task_num_m = task_number_statistic('dataset/ppp', task_num_m)
    x = []
    y = []
    for k, v in task_num_m.items():
        x.append(k)
        y.append(v)
    plt.bar(x, y)
    plt.xlabel('Task number')
    plt.ylabel('Number of RPST')
    plt.show()


def get_rigid(p):
    file_list = os.listdir(p)
    for f in file_list:
        rpst = load_xml(os.path.join(p, f))
        for root in rpst:
            m = {
                'seq': 0,
                'xor': 0,
                'rigid': 0,
                'and': 0,
                'or': 0,
                'loop': 0,
                'task': 0
            }
            dfs(root, m)
            if m['rigid'] > 0:
                print('\''+os.path.join(p, f)+'\'', end=',')


def multi_dir_get_rigid():
    get_rigid('dataset/n_rpst')
    get_rigid('dataset/tested')


def load_gc_read(p):
    arr = []
    with open(p, mode='r', encoding='utf-8') as f:
        for l in f.readlines():
            s = l.split(',')
            if s[1] == 'N':
                arr.append([s[0], s[1], s[2][:-1]])
            else:
                arr.append([s[0], float(s[1]), float(s[2][:-1])])
    return arr


def find_Ns():
    Ns = []
    ps = ['result/Leo_gc_read.txt', 'result/Hen_gc_read.txt', 'result/Goun_gc_read.txt', 'result/BePT_gc_read.txt']
    for p in ps:
        arr = load_gc_read(p)
        for a in arr:
            if a[1] == 'N' and (a[0] not in Ns):
                Ns.append(a[0])
    return Ns


def sort_by_field_gc(field_count_path, gc_read_p):
    field_m = load_fields(field_count_path)
    arr = load_gc_read(gc_read_p)
    m = {
        'Industry': [],
        'Finance': [],
        'Logistics': [],
        'Pharmacy': [],
        'Insurance': [],
        'Education': [],
        'Others': []
    }
    for a in arr:
        if a[1] != 'N':
            m[field_m.get(a[0] + '.xml')].append(a[1])
    for k, v in m.items():
        print(k)
        v = numpy.array(v)
        avg = numpy.average(v, axis=0)
        # print(round_2(v.shape[0] / field_count_m[k]) * 100)
        if v.shape[0] == 0:
            print(round_2(0))
        else:
            print(round_2(avg))


def sort_by_field_gc_without_Ns(field_count_path, gc_read_p):
    field_m = load_fields(field_count_path)
    arr = load_gc_read(gc_read_p)
    m = {
        'Industry': [],
        'Finance': [],
        'Logistics': [],
        'Pharmacy': [],
        'Insurance': [],
        'Education': [],
        'Others': []
    }
    Ns = find_Ns()
    for a in arr:
        if a[0] not in Ns:
            m[field_m.get(a[0] + '.xml')].append(a[1])
    for k, v in m.items():
        print(k)
        v = numpy.array(v)
        avg = numpy.average(v, axis=0)
        # print(round_2(v.shape[0] / field_count_m[k]) * 100)
        if v.shape[0] == 0:
            print(round_2(0))
        else:
            print(round_2(avg))


def convert_rate(field_path, field_count_path, gc_read_p):
    field_m = load_fields(field_path)
    field_count_m = fields__(field_count_path)
    arr = load_gc_read(gc_read_p)
    m = {
        'Industry': 0,
        'Finance': 0,
        'Logistics': 0,
        'Pharmacy': 0,
        'Insurance': 0,
        'Education': 0,
        'Others': 0
    }
    for a in arr:
        if a[1] != 'N':
            m[field_m.get(a[0] + '.xml')] += 1
    for k, v in m.items():
        print(k)
        print(round_2(m[k] / field_count_m[k]) * 100)


def sort_by_field_read(field_count_path, gc_read_p):
    field_m = load_fields(field_count_path)
    arr = load_gc_read(gc_read_p)
    m = {
        'Industry': [],
        'Finance': [],
        'Logistics': [],
        'Pharmacy': [],
        'Insurance': [],
        'Education': [],
        'Others': []
    }
    for a in arr:
        if a[1] != 'N':
            m[field_m.get(a[0] + '.xml')].append(a[2])
    for k, v in m.items():
        print(k)
        v = numpy.array(v)
        avg = numpy.average(v, axis=0)
        # print(round_2(v.shape[0] / field_count_m[k]) * 100)
        if v.shape[0] == 0:
            print(round_2(0))
        else:
            print(round_2(avg))


def sort_by_field_read_without_Ns(field_count_path, gc_read_p):
    field_m = load_fields(field_count_path)
    arr = load_gc_read(gc_read_p)
    m = {
        'Industry': [],
        'Finance': [],
        'Logistics': [],
        'Pharmacy': [],
        'Insurance': [],
        'Education': [],
        'Others': []
    }
    Ns = find_Ns()
    for a in arr:
        if a[0] not in Ns:
            m[field_m.get(a[0] + '.xml')].append(a[2])
    for k, v in m.items():
        print(k)
        v = numpy.array(v)
        avg = numpy.average(v, axis=0)
        # print(round_2(v.shape[0] / field_count_m[k]) * 100)
        if v.shape[0] == 0:
            print(round_2(0))
        else:
            print(round_2(avg))


if __name__ == '__main__':
    multi_dir_get_rigid()
    # fields__('result/fields.txt')
    # multi_dir_statistic()
    # multi_dir_task_number_statistic()
    # multi_dir_get_rigid()
    # field_path = 'result/fields.txt'
    gc_read_p = 'result/HCT_gc_read.txt'
    field_count_path = 'result/fields_96.txt'
    fd = 'result/HCT'
    sort_by_field_read(field_count_path, gc_read_p)
