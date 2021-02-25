from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from joblib import Parallel, delayed
from scipy.spatial import distance
from app.services.CServiceTextUtils import text_stem_sync
from app.model import CTextPairArraysIndexed, CDTOResult

# *******************************************************************************************************
# Класс содержит методы для проверки похожести текстов.                                                 *
# @author Селетков И.П. 2021 0216.                                                                      *
# *******************************************************************************************************


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


def check_sentence(sent1, dict1, dict2, ret, threshold):
    for sent2 in list(dict2.keys()):
        if sent2 not in dict2:
            continue
        simularity_sentences(sent1, dict1, sent2, dict2, ret, threshold)
        if (sent1+sent2) in ret:
            dict2.pop(sent2, None)
            break


def simularity_sentences(sent1, dict1, sent2, dict2, ret, threshold):
    dist = 1-distance.cosine(dict1[sent1], dict2[sent2])
    if dist >= threshold:
        ret[sent1+sent2] = CDTOResult(k=dist, s1=sent1, s2=sent2)


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
    ret = dict()

    Parallel(n_jobs=-1, require='sharedmem')(
         delayed(check_sentence)
         (sent, dict1, dict2, ret, threshold)
         for sent in dict1.keys()
    )

    return list(ret.values())


def treat_sent_indexed(par, vectorizer, dict1):
    dict1[par.text] = [par.n, vectorizer.transform([par.text]).toarray()]


def check_sentence_indexed(sent1, dict1, dict2, ret, threshold):
    for sent2 in list(dict2.keys()):
        if sent2 not in dict2:
            continue
        simularity_sentences_indexed(sent1, dict1, sent2, dict2, ret, threshold)
        if (sent1+sent2) in ret:
            dict2.pop(sent2, None)
            break


def simularity_sentences_indexed(sent1, dict1, sent2, dict2, ret, threshold):
    dist = 1-distance.cosine(dict1[sent1][1], dict2[sent2][1])
    if dist >= threshold:
        ret[sent1+sent2] = CDTOResult(k=dist, n1=dict1[sent1][0], s1=sent1, n2=dict2[sent2][0], s2=sent2)


# ***************************************************************************************************
# Расчёт похожести двух текстов по косинусной метрике.                                              *
# using `TF-IDF` & `Cosine-similarity`                                                              *
# @param sents1 - входной текст в виде массива предобработанных предложений.                        *
# @param sents2 - входной текст в виде массива предобработанных предложений.                        *
# @return степень похожести текстов.                                                                *
# ***************************************************************************************************
async def similarity_cleared_indexed(data: CTextPairArraysIndexed, threshold=0.7):
    dict1 = dict()
    dict2 = dict()

    sents1 = [par.text for par in data.text1]
    sents2 = [par.text for par in data.text1]

    vectorizer = TfidfVectorizer()
    vectorizer.fit(sents1 + sents2)

    Parallel(n_jobs=-1, require='sharedmem')(
        delayed(treat_sent_indexed)(par, vectorizer, dict1)
        for par in data.text1
    )
    Parallel(n_jobs=-1, require='sharedmem')(
        delayed(treat_sent_indexed)(par, vectorizer, dict2)
        for par in data.text2
    )

    ret = dict()

    Parallel(n_jobs=-1, require='sharedmem')(
         delayed(check_sentence_indexed)
         (sent1, dict1, dict2, ret, threshold)
         for sent1 in dict1.keys()
    )

    return list(ret.values())
