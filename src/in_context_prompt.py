from call_llm import prompt_llm
from utils import load_txt, save_as_txt
import os
import metric


def in_context(model, pre, temperature, rpst, prompt):
    prompt_ = prompt + rpst
    swd = prompt_llm(model, pre, prompt_, temperature)
    return swd


def main_process():
    rpst_dir = 'dataset/n_rpst'
    ground_truth_dir = 'dataset/swd'
    prompt_path = 'prompt/in_context/prompt.txt'
    save_dir = 'result/ICL'
    model = 'gpt-3.5-turbo'
    pre = {'role': 'system', 'content': 'You are an assistant who follows instructions.'}
    temperature = 0
    prompt = load_txt(prompt_path)
    file_list = os.listdir(rpst_dir)

    # test
    # file = 'dataset/n_rpst/2748_1dbaeb7a01b9419aa67a4c1ed5eb7153.xml'
    # rpst = load_txt(file)
    # for root in rpst:
    #     swd = in_context(model, pre, temperature, rpst, prompt)
    #     gc = metric.grammar_correctness(swd)
    i = 0
    for file in range(len(file_list)):
        rpst = load_txt(os.path.join(rpst_dir, file_list[file + i]))
        print('{}'.format(file_list[file + i][:-4]), end=',')
        # ground_truth = load_txt(os.path.join(ground_truth_dir, file[:-3] + '.txt'))
        swd = in_context(model, pre, temperature, rpst, prompt)
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
        # print('rpst: {file}, grammar correctness: {gc}, readability: {read}, meteor: {mt}, rouge: {rg}, bert score: {bs}, sentence bert: {sbert}'.format(file=file, gc=gc, read=read, mt=mt, rg=rg, bs=bs, sbert=sbert))
        print('{},{}'.format(gc, read))
        save_as_txt(os.path.join(save_dir, file_list[file + i][:-4] + '.txt'), swd)


if __name__ == '__main__':
    main_process()
