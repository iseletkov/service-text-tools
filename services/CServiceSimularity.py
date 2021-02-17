from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from services.CServiceTextUtils import text_stem_sync
from pydantic import BaseModel


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


# ***************************************************************************************************
# Расчёт похожести двух текстов по косинусной метрике.                                              *
# using `TF-IDF` & `Cosine-similarity`                                                              *
# @param sents1 - входной текст в виде массива предобработанных предложений.                        *
# @param sents2 - входной текст в виде массива предобработанных предложений.                        *
# @return степень похожести текстов.                                                                *
# ***************************************************************************************************
async def similarity_cleared(sents1, sents2):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(sents1+sents2)

    arr = []
    for i, sent1 in enumerate(sents1, start=0):

        sent1 = "".join(sents1[i])
        for j, sent2 in enumerate(sents2, start=0):
            sent2 = "".join(sents2[j])
            tfidf = vectorizer.transform([sent1, sent2])
            tfidf = (tfidf * tfidf.T).A[0, 1]
            if tfidf > 0.2:
                arr.append(CResult(tfidf, i, sent1, j, sent2))
    return arr

