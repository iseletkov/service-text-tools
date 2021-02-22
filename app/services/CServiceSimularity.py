from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from app.services.CServiceTextUtils import text_stem_sync
from joblib import Parallel, delayed
# *******************************************************************************************************
# Класс содержит методы для проверки похожести текстов.                                                 *
# @author Селетков И.П. 2021 0216.                                                                      *
# *******************************************************************************************************

class CResult:
    def __init__(self, simularity, num1, sent1, num2, sent2):
        self.k = simularity
        self.n1 = num1
        self.s1 = sent1
        self.n2 = num2
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


def join_sent(sent):
    return "".join(sent)


def join_sent_num(sent, i):
    return [join_sent(sent), i]

# ***************************************************************************************************
# Расчёт похожести двух текстов по косинусной метрике.                                              *
# using `TF-IDF` & `Cosine-similarity`                                                              *
# @param sents1 - входной текст в виде массива предобработанных предложений.                        *
# @param sents2 - входной текст в виде массива предобработанных предложений.                        *
# @return степень похожести текстов.                                                                *
# ***************************************************************************************************
async def similarity_cleared(sents1, sents2):
    threshold = 0.7
    vectorizer = TfidfVectorizer()
    vectorizer.fit(sents1 + sents2)

    sents1 = Parallel(n_jobs=-1, require='sharedmem')(
        delayed(join_sent)(sent)
        for sent in sents1
    )
    sents2 = Parallel(n_jobs=-1, require='sharedmem')(
        delayed(join_sent)(sent)
        for sent in sents2
    )
    sentsn2 = Parallel(n_jobs=-1, require='sharedmem')(
        delayed(join_sent_num)(sent, i)
        for i, sent in enumerate(sents2, start=0)
    )

    len1 = len(sents1)
    len2 = len(sents2)

    arr = [[0] * len2 for _ in range(len1)]

    Parallel(n_jobs=-1, require='sharedmem')(
         delayed(check_sentence)
         (sent, i, sentsn2, vectorizer, arr, threshold)
         for i, sent in enumerate(sents1, start=0)
    )

    ret = []
    for i in range(len1):
        for j in range(len2):
            if arr[i][j] >= threshold:
                ret.append(CResult(arr[i][j], i, sents1[i], j, sents2[j]))

    return ret


def check_sentence(sent1, i, sentsn2, vectorizer, arr, threshold):
    for sent2 in sentsn2:
        simularity_sentences(sent1, i, sent2[0], sent2[1], vectorizer, arr)
        if arr[i][sent2[1]] >= threshold:
            sentsn2.remove(sent2)
            break
    return arr


def simularity_sentences(sent1, i, sent2, j, vectorizer, arr):
    tfidf = vectorizer.transform([sent1, sent2])
    arr[i][j] = (tfidf * tfidf.T).A[0, 1]
