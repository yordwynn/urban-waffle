import pickle
from typing import List, Dict


def parse_model(file_path: str) -> Dict[str, List[float]]:
    res = {}

    with open(file_path) as f:
        (words, dim) = map(lambda x: int(x), f.readline().split(' '))

        for line in f:
            items = line.split(' ')
            word = items[0]
            embeding = parse_embeding(items[1:])

            if (len(embeding) == dim):
                res[word] = embeding

    return res if len(res) == words else {}


def load_file(file_path: str) -> List[str]:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data

def parse_embeding(items: List[str]) -> List[float]:
    return list(map(lambda x: float(x), items))