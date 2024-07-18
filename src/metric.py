import numpy as np
import textstat
import language_tool_python
from nltk.tokenize import sent_tokenize
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from bleurt import score


def round_2(number):
    return round(number, 2)


def round_4(number):
    return round(number, 4)


my_tool = language_tool_python.LanguageTool('en-US')


def grammar_correctness(swd):
    if swd != '':
        gc = 1 - (len(my_tool.check(swd)) / len(sent_tokenize(swd)))
        return round_2(gc)
    else:
        return 'N'


def readability_score(swd):
    if swd != '':
        flesch_kincaid = textstat.flesch_kincaid_grade(swd)
        gunning_fog = textstat.gunning_fog(swd)
        coleman_liau = textstat.coleman_liau_index(swd)
        dale_chall = textstat.dale_chall_readability_score(swd)
        r = np.array([flesch_kincaid, gunning_fog, coleman_liau, dale_chall])
        return round_2(np.average(r))
    else:
        return 'N'


def readability_score_(swd):
    if swd != '':
        flesch_kincaid = textstat.flesch_kincaid_grade(swd)
        gunning_fog = textstat.gunning_fog(swd)
        coleman_liau = textstat.coleman_liau_index(swd)
        dale_chall = textstat.dale_chall_readability_score(swd)
        r = np.array([flesch_kincaid, gunning_fog, coleman_liau, dale_chall])
        return [round_2(flesch_kincaid), round_2(gunning_fog), round_2(coleman_liau), round_2(dale_chall),
                round_2(np.average(r))]
    else:
        return 'N'


def meteor(swd, ground_truth):
    if swd != '':
        return round_4(meteor_score([ground_truth.split()], swd.split()))
    else:
        return round_4(0)


rg = Rouge()


def rouge(swd, ground_truth):
    if swd != '':
        scores = rg.get_scores(swd, ground_truth)
        rg1 = scores[0]['rouge-1']['f']
        rg2 = scores[0]['rouge-2']['f']
        rgl = scores[0]['rouge-l']['f']
        return round_4((rg1 + rg2 + rgl) / 3)
    else:
        return round_4(0)


scorer = BERTScorer(model_type=r'G:\models\bert-score', num_layers=6)


def bert_score(swd, ground_truth):
    if swd != '':
        P, R, bs = scorer.score([swd], [ground_truth])
        return round_4(bs.item())
    else:
        return round_4(0)


sbert = SentenceTransformer(r'G:\models\all-mpnet-base-v2')


def s_bert(swd, ground_truth):
    if swd != '':
        embeds = sbert.encode([ground_truth, swd])
        return round_4(1 - cosine(embeds[0], embeds[1]))
    else:
        return round_4(0)


checkpoint = 'lib/bleurt-master/bleurt/BLEURT-20'
scorer_ble = score.BleurtScorer(checkpoint)


def BLEURT(swd, ground_truth):
    if swd != '':
        scores = scorer_ble.score(references=[ground_truth], candidates=[swd])
        return round_4(scores[0])
    else:
        return round_4(0)
