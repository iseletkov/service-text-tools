from fastapi import APIRouter
from typing import Optional
from app.services import CServiceTextUtils
from app.services import CServiceSimularity
from app.model import CDTOText, CDTOTextIndexed, CTextPair, CTextPairArrays, CDTOParagraph, CTextPairArraysIndexed

router = APIRouter()


# *******************************************************************************************************
# Контроллер содержит обработку запросов на текстовые операции.                                         *
# @author Селетков И.П. 2019 1118.                                                                      *
# *******************************************************************************************************

# *******************************************************************************************************
# Выполнение лемматизации.                                                                              *
# *******************************************************************************************************
@router.post("/lemmatization")
async def lemmatization(data: CDTOText):
    return await CServiceTextUtils.text_lemmatization(data.text)


# *******************************************************************************************************
# Выполнение стемминга.                                                                                 *
# *******************************************************************************************************
@router.post("/stemming")
async def stemming(data: CDTOText):
    return await CServiceTextUtils.text_stem(data.text)


# *******************************************************************************************************
# Выполнение стемминга.                                                                                 *
# *******************************************************************************************************
@router.post("/stemming_sentences")
async def stemming_sentences(data: CDTOText):
    return await CServiceTextUtils.stemming_sentences(data.text)


# *******************************************************************************************************
# Выполнение стемминга.                                                                                 *
# *******************************************************************************************************
@router.post("/stemming_sentences_indexed")
async def stemming_sentences_indexed(data: CDTOTextIndexed):
    return await CServiceTextUtils.stemming_sentences_indexed(data)


@router.post("/test")
async def test(data: CDTOParagraph):
    return data


# *******************************************************************************************************
# Выполнение лемматизации.                                                                              *
# *******************************************************************************************************
@router.post("/clear")
async def clear(data: CDTOText):
    words = await CServiceTextUtils.clear_text(data.text)
    return " ".join(words)


# *******************************************************************************************************
# Выполнение сравнения двух текстов.                                                                    *
# *******************************************************************************************************
@router.post("/simularity")
async def simularity(data: CTextPair):
    return await CServiceSimularity.similarity(data.text1, data.text2)


# *******************************************************************************************************
# Выполнение сравнения двух предварительно подготовленных текстов.                                      *
# *******************************************************************************************************
@router.post("/simularity_cleared")
async def simularity_cleared(
        data: CTextPairArrays,
        threshold: Optional[float] = 0.7
):
    return await CServiceSimularity.similarity_cleared(data.text1, data.text2, threshold)


# *******************************************************************************************************
# Выполнение сравнения двух предварительно подготовленных текстов.                                      *
# *******************************************************************************************************
@router.post("/simularity_cleared_indexed")
async def simularity_cleared_indexed(
        data: CTextPairArraysIndexed,
        threshold: Optional[float] = 0.7
):
    return await CServiceSimularity.similarity_cleared_indexed(data, threshold)
