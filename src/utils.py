import os
import xml.etree.ElementTree as ET
import pm4py


def load_txt(path):
    with open(path, mode='r', encoding='utf-8') as f:
        prompt = f.read()
        return prompt


def save_as_txt(path, string):
    with open(path, mode='w', encoding='utf-8') as f:
        f.write(string)


def load_xml(path):
    tree = ET.parse(path)
    return tree.getroot()


def load_dataset(dir):
    files = os.listdir(dir)
    rpst_list = []
    for file in files:
        rpst_list.append(load_xml(file))
    return rpst_list


def fill_file(json_dir, txt_dir):
    json_files = os.listdir(json_dir)
    txt_files = os.listdir(txt_dir)
    for jf in json_files:
        exist = False
        for tf in txt_files:
            if jf[:-5] == tf[:-4]:
                exist = True
                break
        if not exist:
            os.open(jf[:-5] + '.txt', os.O_CREAT)


def load_file_names(p):
    fs = os.listdir(p)
    return fs


def check():
    fs = load_file_names('dataset/n_rpst')
    fs.extend(load_file_names('dataset/tested'))
    bpmns = load_file_names('dataset/bpmn')
    for f in fs:
        exist = False
        for b in bpmns:
            if f[:-4] == b[:-5]:
                exist = True
                break
        if not exist:
            print(f)


def convert_bpmn_to_petrinet(bpmn_dir, save_dir):
    file_list = os.listdir(bpmn_dir)
    for file in file_list:
        bpmn = pm4py.read_bpmn(os.path.join(bpmn_dir, file))
        net, im, fm = pm4py.convert_to_petri_net(bpmn)
        pm4py.write.write_pnml(net, im, fm, os.path.join(save_dir, file[:-4] + 'pnml'))


if __name__ == '__main__':
    convert_bpmn_to_petrinet('dataset/bpmn', 'G:\huquanzhou\BePT\datasets\dataset')
