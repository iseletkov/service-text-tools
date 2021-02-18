from fastapi import APIRouter
from app.services import CServiceTextUtils
from app.services import CServiceSimularity

from pydantic import BaseModel

router = APIRouter()


# *******************************************************************************************************
# Контроллер содержит обработку запросов на текстовые операции.                                         *
# @author Селетков И.П. 2019 1118.                                                                      *
# *******************************************************************************************************
class CData(BaseModel):
    text: str


class CTextPair(BaseModel):
    text1: str
    text2: str


class CTextPairArrays(BaseModel):
    text1: list
    text2: list


# *******************************************************************************************************
# Выполнение лемматизации.                                                                              *
# *******************************************************************************************************
@router.post("/lemmatization")
async def lemmatization(data: CData):
    return await CServiceTextUtils.text_lemmatization(data.text)


# *******************************************************************************************************
# Выполнение стемминга.                                                                                 *
# *******************************************************************************************************
@router.post("/stemming")
async def stemming(data: CData):
    return await CServiceTextUtils.text_stem(data.text)


# *******************************************************************************************************
# Выполнение стемминга.                                                                                 *
# *******************************************************************************************************
@router.post("/stemming_sentences")
async def stemming_sentences(data: CData):
    return await CServiceTextUtils.stemming_sentences(data.text)


# *******************************************************************************************************
# Выполнение лемматизации.                                                                              *
# *******************************************************************************************************
@router.post("/clear")
async def clear(data: CData):
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
async def simularity_cleared(data: CTextPairArrays):
    return await CServiceSimularity.similarity_cleared(data.text1, data.text2)
