from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from app.services.CServiceTextUtils import text_stem_sync
from joblib import Parallel, delayed
from scipy.spatial import distance
# *******************************************************************************************************
# Класс содержит методы для проверки похожести текстов.                                                 *
# @author Селетков И.П. 2021 0216.                                                                      *
# *******************************************************************************************************

class CResult:
    def __init__(self, simularity, sent1, sent2):
        self.k = simularity
        self.s1 = sent1
        self.s2 = sent2


# ***************************************************************************************************
# Расчёт похожести двух текстов по косинусной метрике.                                              *
# using `TF-IDF` & `Cosine-similarity`                                                              *
# @param text1 - входной текст.                                                                     *
# @param text2 - входной текст.                                                                     *
# @return степень похожести текстов.                                                                *
# ***************************************************************************************************
async def similarity(text1, text2):
    sents1 = sent_tokenize(text1, "russian")
    sents2 = sent_tokenize(text2, "russian")

    sents1 = [await text_stem_sync(sent) for sent in sents1]
    sents2 = [await text_stem_sync(sent) for sent in sents2]

    return await similarity_cleared(sents1, sents2)


def treat_sent(sent, vectorizer, dict1):
    dict1["".join(sent)] = vectorizer.transform([sent]).toarray()


# ***************************************************************************************************
# Расчёт похожести двух текстов по косинусной метрике.                                              *
# using `TF-IDF` & `Cosine-similarity`                                                              *
# @param sents1 - входной текст в виде массива предобработанных предложений.                        *
# @param sents2 - входной текст в виде массива предобработанных предложений.                        *
# @return степень похожести текстов.                                                                *
# ***************************************************************************************************
async def similarity_cleared(sents1, sents2, threshold=0.7):
    dict1 = dict()
    dict2 = dict()

    vectorizer = TfidfVectorizer()
    vectorizer.fit(sents1 + sents2)

    Parallel(n_jobs=-1, require='sharedmem')(
        delayed(treat_sent)(sent, vectorizer, dict1)
        for sent in sents1
    )
    Parallel(n_jobs=-1, require='sharedmem')(
        delayed(treat_sent)(sent, vectorizer, dict2)
        for sent in sents2
    )
    sentsn2 = dict2.copy()

    ret = dict()

    Parallel(n_jobs=-1, require='sharedmem')(
         delayed(check_sentence)
         (sent, dict1, sentsn2, ret, threshold)
         for sent in dict1.keys()
    )

    return list(ret.values())


def check_sentence(sent1, dict1, sentsn2, ret, threshold):
    for sent2 in list(sentsn2.keys()):
        if sent2 not in sentsn2:
            continue
        simularity_sentences(sent1, dict1, sent2, sentsn2, ret, threshold)
        if (sent1+sent2) in ret:
            sentsn2.pop(sent2, None)
            break


def simularity_sentences(sent1, dict1, sent2, sentsn2, ret, threshold):
    dist = 1-distance.cosine(dict1[sent1], sentsn2[sent2])
    if dist >= threshold:
        ret[sent1+sent2] = CResult(dist, sent1, sent2)

