from typing import List
from pydantic import BaseModel


class CDTOText(BaseModel):
    text: str


class CDTOParagraph(BaseModel):
    n: int
    text: str


class CDTOTextIndexed(BaseModel):
    pars: List[CDTOParagraph]


class CTextPair(BaseModel):
    text1: str
    text2: str


class CTextPairArrays(BaseModel):
    text1: list
    text2: list


class CTextPairArraysIndexed(BaseModel):
    text1: List[CDTOParagraph]
    text2: List[CDTOParagraph]


class CDTOResult(BaseModel):
    k:  float
    n1: int
    s1: str
    n2: int
    s2: str
