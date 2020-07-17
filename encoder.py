import pickle
import numpy as np
import pymorphy2, re
from typing import List, Dict


def parse_model(file_path: str) -> Dict[str, List[float]]:
    res = {}

    with open(file_path) as f:
        (words, dim) = map(lambda x: int(x), f.readline().split(' '))

        for line in f:
            items = line.split(' ')
            word = items[0]
            embeding = parse_embeding(items[1:])

            if (embeding.size == dim):
                res[word] = embeding

    return res if len(res) == words else {}


def load_from_pickle(file_path: str) -> List[str]:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data

def save_to_pickle(data, file_path: str):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def parse_embeding(items: List[str]) -> List[float]:
    return np.fromiter(map(lambda x: float(x), items), dtype = float)

def check_in_w2v(word: str, model: Dict[str, np.ndarray], ma) -> str:
    pos = {'ADJF':'ADJ', 'ADJS':'ADJ', 'COMP':'ADJ', 'ADVB':'ADV', 'INFN':'VERB', 'NUMR':'NUM', \
       'PRTF':'VERB', 'PRTS':'VERB', 'GRND':'VERB', 'NPRO':'PRON', 'CONJ':'CCONJ'}

    part_of_speech = str(ma.parse(word)[0].tag.POS).upper()
    try:
      part_of_speech = pos[part_of_speech]
    except:
      pass

    word += "_" + part_of_speech
    if word in model:
        return word
    else:
        return ""

def clean_text(text: str, model: Dict[str, np.ndarray]) -> str:
    ma = pymorphy2.MorphAnalyzer()

    text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ").replace("«", '').replace("»", '').replace("…", '')
    text = text.lower()
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text) 
    text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)
    text = " ".join(ma.parse(word)[0].normal_form for word in text.split())
    text = ' '.join(word for word in text.split() if len(word)>3)
    all_words = text.split()
    all_correct_words = ''

    for word in all_words:
        new_word = check_in_w2v(word, model, ma)
        if new_word != '':
            if all_correct_words=='':
                all_correct_words += new_word
            else:
                all_correct_words += ' '+new_word
    return all_correct_words

def encode_text(text: str, model: Dict[str, np.ndarray], shape: (int, int)) -> np.ndarray:
    res = np.zeros(shape)
    for i, word in enumerate(text.split()):
        res[i,:] = model[word]

    return res