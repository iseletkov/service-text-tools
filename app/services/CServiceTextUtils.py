# -*- coding: utf8 -*-

import re
import pymorphy2
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from app.services.CServicePorter import stem
from app.model.CModel import CDTOParagraph, CDTOTextIndexed
# *******************************************************************************************************
# Файл содержит реализацию текстовых операций.                                                          *
# @author @kocter. 2019 1003.                                                                           *
# *******************************************************************************************************

auxiliary_words = [
    "для", "которая", "только", "то", "нас", "них", "нам", "вам", "всех", "нем", "нему", "аля", "без",
    "безведома", "безо", "благодаря", "близ",
    "в", "ввиде", "вблизи", "ввиду", "вглубь", "вдогон", "вдоль", "взамен", "включая", "вкруг", "вместо",
    "вне",
    "внизу", "внутри", "внутрь", "во", "вовнутрь", "возле", "вокруг", "вопреки", "вослед", "впереди",
    "вразрез",
    "вроде", "вслед", "вследствие", "встречу", "выключая", "для", "для-ради", "до", "за", "замест",
    "заместо",
    "из", "из-за", "из-под", "из-подо", "изнутри", "изо", "исключая", "к", "касаемо", "касательно", "ко",
    "кончая", "кроме", "кругом", "меж", "между", "мимо", "на", "наверху", "навроде", "навстречу", "надо",
    "назад", "назади", "накануне", "наместо", "наперекор", "наперерез", "наперехват", "наподобие",
    "наподобье",
    "напротив", "насупротив", "насчёт", "несмотря", "несмотря на", "ниже", "о", "об", "обо", "обок",
    "обочь",
    "около", "окрест", "окроме", "окромя", "округ", "опосля", "опричь", "от", "относительно", "ото",
    "перед",
    "передо", "по", "по-за", "по-над", "по-под", "повдоль", "поверх", "под", "подле", "подо", "подобно",
    "позади", "позадь", "позднее", "помимо", "поперёд", "поперёк", "порядка", "посверху", "посереди",
    "посередине", "посерёдке", "посередь", "после", "посреди", "посредине", "посредством", "пред",
    "предо",
    "преж", "прежде", "при", "про", "промеж", "промежду", "против", "противно", "противу", "путём",
    "ради",
    "с", "сверх", "сверху", "свыше", "середи", "середь", "сзади", "скрозь", "снизу", "со", "согласно",
    "спустя", "среди", "средь", "сродни", "супротив", "у", "через", "черезо", "чрез", "кось", "ста", "а",
    "авось", "авось-либо", "ага", "адьё", "аж", "ажно", "аиньки", "аминь", "ан", "аск", "аушки", "аюшки",
    "б", "бишь", "будто", "буквально", "бы", "бывает", "было", "ведь", "верно", "вероятно", "вестимо",
    "вишь",
    "во", "вон", "вот", "вот-вот", "вроде", "вряд", "всё", "всё-таки", "всего", "го", "да", "давай",
    "давай-ка",
    "даже", "дак", "де", "действительно", "дескать", "добре", "дык", "едва", "если", "ещё", "ж", "же",
    "замётано", "и", "или", "иль", "именно", "имхо", "ин", "инда", "индо", "ино", "ить", "ишь", "ка",
    "кажется",
    "кайнэ", "как", "как бы", "конечно", "куда", "ладно", "ладушки", "ли", "лих", "лишь", "лучше", "ль",
    "мол",
    "на", "на-тка", "навроде", "накось", "накося", "наоборот", "не", "не-а", "небось", "нет", "нет-нет",
    "нету",
    "неуж", "неужели", "неужли", "неужто", "ни", "никак", "ничего", "ништо", "ну", "ну-ну", "ну-с",
    "отколь",
    "откуда", "откудова", "отож", "очевидно", "поди", "пожалуй", "пожалуйста", "пока", "с", "так",
    "таки",
    "того", "то-то", "тоже", "также", "уж", "уже", "ужели", "хоть", "хотя", "что", "что-то", "чтоб",
    "чтобы",
    "чтой-то", "а", "абы", "аж", "ажно", "ай", "аки", "ако", "али", "аль", "ан", "аще", "благо", "бо",
    "буде",
    "будто", "ведь", "впрочем", "всё", "всё-таки", "где", "где-то", "да", "дабы", "даже", "докуда",
    "дотоле",
    "егда", "едва", "еже", "ежели", "ежель", "если", "ж", "же", "зане", "занеже", "зато", "зачем",
    "значит",
    "и", "или", "ибо", "или", "иль", "именно", "имже", "инако", "иначе", "инда", "ино", "итак", "кабы",
    "как",
    "каков", "какой", "ковда", "ковды", "когда", "когды", "коли", "коль", "который", "куда", "ли",
    "либо",
    "лишь", "ль", "настолько", "нежели", "незомь", "ни", "ниже", "нижли", "но", "обаче", "однако",
    "одначе",
    "отколь", "откуда", "откудова", "оттого", "отчего", "поелику", "пока", "покамест", "покаместь",
    "покеда",
    "поколева", "поколику", "поколь", "покуль", "покуля", "понеже", "поскольку", "пота", "потолику",
    "потому",
    "почём", "почему", "правда", "преж", "притом", "причём", "просто", "пускай", "пусть", "равно", "раз",
    "разве", "ровно", "тож", "тоже", "только", "хоть", "хотя", "чем", "чи", "что", "чтоб", "чтобы",
    "чуть",
    "штоб", "штобы", "яко", "якобы", "я", "мы", "ты", "вы", "он", "она", "оно", "они", "себя", "мой",
    "моя",
    "мое", "мои", "наш", "наша", "наше", "наши", "твой", "твоя", "твое", "твои", "ваш", "ваша", "ваше",
    "ваши",
    "его", "ее", "их", "кто", "что", "какой", "каков", "чей", "который", "сколько", "где", "когда",
    "куда",
    "зачем", "этот", "тот", "такой", "таков", "тут", "здесь", "сюда", "туда", "оттуда", "отсюда",
    "тогда",
    "поэтому", "это", "этим", "этому", "этом", "этих", "сей", "оный", "затем", "столько", "весь",
    "всякий",
    "все", "сам", "самый", "каждый", "любой", "другой", "иной", "всяческий", "всюду", "везде", "всегда",
    "никто", "ничто", "некого", "нечего", "никакой", "ничей", "некто", "нечто", "некий", "некоторый",
    "несколько", "кое-кто", "кое-где", "кое-что", "кое-куда", "какой-либо", "сколько-нибудь", "свой",
    "куда-нибудь", "зачем-нибудь", "чей-либо", "один", "два", "три", "четыре", "пять", "шесть", "семь",
    "восемь",
    "девять", "ноль", "эту", "сих", "нашу", "тому", "там", "таких", "тех", "над", "вас", 'который',
    'один', "такой", 'тот',
    'этот', 'другой', 'два', 'два', 'каждый', 'себя', 'тем', 'тем', 'он', 'еще', 'она', 'какой', 'том',
    'что', 'теперь',
    'сам', 'теперь', 'сам', 'ваш', 'тд'
]

auxiliary_words_porter = [
    "для", "тольк", "то", "ла", "ег", "с", "не", "нас", "них", "нам", "вам", "всех", "н", "на", "и", "нему", "ал",
    "без",
    "безведом", "безо", "благодар", "близ", "в", "ввид", "вблиз", "ввиду", "вглуб", "вдогон", "вдол", "взамен",
    "включ", "вкруг", "вместо", "вн", "внизу", "внутр", "внутр", "во", "вовнутр", "возл", "вокруг", "вопрек",
    "вослед", "вперед", "вразрез", "врод", "вслед", "вследств", "встречу", "выключ", "дл", "для-рад", "до", "з",
    "замест", "заместо", "из", "из-з", "из-под", "из-подо", "изнутр", "изо", "исключ", "к", "касаемо",
    "касательно", "ко", "конч", "кром", "круг", "меж", "между", "мимо", "н", "наверху", "наврод", "навстречу",
    "надо", "назад", "назад", "наканун", "наместо", "наперекор", "наперерез", "наперехват", "наподоб",
    "наподоб", "напрот", "насупрот", "насчет", "несмотр", "несмотря", "н", "ниж", "о", "об", "обо", "обок",
    "обоч", "около", "окрест", "окром", "окром", "округ", "опосл", "оприч", "от", "относительно", "ото",
    "перед", "передо", "по", "по-з", "по-над", "по-под", "повдол", "поверх", "под", "подл", "подо", "подобно",
    "позад", "позад", "поздн", "помимо", "поперед", "поперек", "порядк", "посверху", "посеред", "посередин",
    "посередк", "посеред", "посл", "посред", "посредин", "посредств", "пред", "предо", "преж", "прежд", "пр",
    "про", "промеж", "промежду", "прот", "противно", "противу", "пут", "рад", "с", "сверх", "сверху", "свыш",
    "серед", "серед", "сзад", "скроз", "снизу", "со", "согласно", "спуст", "сред", "сред", "сродн", "супрот",
    "у", "через", "черезо", "чрез", "ко", "ст", "", "аво", "авось-либо", "аг", "ад", "аж", "ажно", "аиньк",
    "амин", "а", "аск", "аушк", "аюшк", "б", "б", "будто", "буквально", "б", "быва", "было", "вед", "верно",
    "вероятно", "вестимо", "в", "во", "вон", "вот", "вот-вот", "врод", "вряд", "вс", "все-так", "вс", "го", "д",
    "дава", "давай-к", "даж", "дак", "д", "действительно", "деска", "добр", "дык", "едв", "есл", "ещ", "ж", "ж",
    "замета", "", "", "ил", "именно", "имхо", "ин", "инд", "индо", "ино", "", "", "к", "кажет", "кайнэ", "как",
    "как", "б", "конечно", "куд", "ладно", "ладушк", "л", "лих", "л", "лучш", "ль", "мол", "н", "на-тк",
    "наврод", "нако", "нако", "наоборот", "н", "не-", "небо", "нет", "нет-нет", "нету", "неуж", "неужел",
    "неужл", "неужто", "н", "никак", "нич", "ништо", "ну", "ну-ну", "ну-с", "откол", "откуд", "откудов", "отож",
    "очевидно", "под", "пожал", "пожалуйст", "пок", "с", "так", "так", "т", "то-то", "тож", "такж", "уж", "уж",
    "ужел", "хот", "хот", "что", "что-то", "чтоб", "чтоб", "чтой-то", "", "аб", "аж", "ажно", "а", "ак", "ако",
    "а", "ал", "а", "ащ", "благо", "бо", "буд", "будто", "вед", "впроч", "вс", "все-так", "гд", "где-то", "д",
    "даб", "даж", "докуд", "дотол", "егд", "едв", "еж", "ежел", "ежел", "есл", "ж", "ж", "зан", "занеж", "зато",
    "зач", "значит", "", "", "ибо", "", "ил", "именно", "имж", "инако", "инач", "инд", "ино", "итак", "каб",
    "как", "как", "как", "ковд", "ковд", "когд", "когд", "кол", "кол", "котор", "куд", "л", "либо", "л", "ль",
    "настолько", "нежел", "незом", "н", "ниж", "нижл", "но", "обач", "однако", "однач", "откол", "откуд",
    "откудов", "отт", "отч", "поелику", "пок", "покамест", "покамест", "покед", "поколев", "поколику", "покол",
    "покул", "покул", "понеж", "поскольку", "пот", "потолику", "потому", "поч", "почему", "правд", "преж",
    "прит", "прич", "просто", "пуска", "пуст", "равно", "раз", "разв", "ровно", "тож", "тож", "только", "хот",
    "хот", "ч", "ч", "что", "чтоб", "чтоб", "чут", "штоб", "штоб", "яко", "якоб", "", "м", "т", "в", "он", "он",
    "оно", "он", "себ", "м", "мо", "м", "мо", "наш", "наш", "наш", "наш", "тв", "тво", "тв", "тво", "ваш",
    "ваш", "ваш", "ваш", "", "", "их", "кто", "что", "как", "как", "ч", "котор", "сколько", "гд", "когд", "куд",
    "зач", "этот", "тот", "так", "так", "тут", "зд", "сюд", "туд", "оттуд", "отсюд", "тогд", "поэтому", "это",
    "эт", "этому", "эт", "этих", "с", "он", "зат", "столько", "в", "всяк", "вс", "сам", "сам", "кажд", "люб",
    "друг", "ин", "всяческ", "всюду", "везд", "всегд", "никто", "ничто", "нек", "неч", "никак", "нич", "некто",
    "нечто", "нек", "некотор", "несколько", "кое-кто", "кое-гд", "кое-что", "кое-куд", "какой-либо",
    "сколько-нибуд", "св", "куда-нибуд", "зачем-нибуд", "чей-либо", "один", "дв", "тр", "четыр", "пя", "шест",
    "сем", "восем", "девя", "нол", "нашу", "эту", "сих", "тому", "там", "таких", "тех", "над", "вас"
]

morph = pymorphy2.MorphAnalyzer()


# *******************************************************************************************************
# Удаление всех вспомогательных символов.                                                               *
# @param str - входная строка.                                                                          *
# @return строка без вспомогательных символов.                                                          *
# *******************************************************************************************************
async def filter_symbols(text):
    text = re.sub(r"[.!,:=\\//[]{}()+-<!;?(<)>—""%#@&'']", "", text)
    text = re.sub(r"[«]", "", text)
    text = re.sub(r"[»]", "", text)
    text = re.sub(r"[<>[=/_—]", "", text)
    text = re.sub(r"[-.:')!,]", "", text)
    text = re.sub(r"[%;?&(-]", "", text)
    text = re.sub(r"[1234567890]", "",text)
    text = re.sub(r"[\"]", "", text)
    text = re.sub(r"[qwertyuiopasdfghjklzxcvbnm]", "", text)  #Удаляет весь английский
    text = re.sub(r'\s+', ' ', text)  # Удаляет лишние пробелы появившиеся после удаления символов и цифр
    return text


# *******************************************************************************************************
# Добавление новых слов в словарь.                                                                      *
# @param words - слова из текста.                                                                       *
# @return актуализированный словарь.                                                                    *
# *******************************************************************************************************
# def add_to_dictionary(words):
#     # Сюда нужно подцепить словарь
#     dictionary = ['запущ', 'космос', 'спутник', 'связ', 'прав', 'поступ', 'циркулир', 'высот', 'мил']
#     for i in words:
#         if i not in dictionary:
#             dictionary.append(i)
#     return dictionary


def clear_text_sync(text):
    # split into words
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('russian'))
    words = [w for w in words if w not in stop_words]
    return words


async def clear_text(text):
    return clear_text_sync(text)


# *******************************************************************************************************
# Чистка текста с помощью метода Портера.                                                               *
# @param text - исходный текст.                                                                         *
# @return обработанный текст.                                                                           *
# *******************************************************************************************************
def tokenize_stem_sync(text):
    words = clear_text_sync(text)
    words = list(map(lambda word: stem(word), words))  # Прохождение стеммером Портера
    words = list(filter(lambda word: len(word) > 3, words))  # Удаление всего, что короче 2 символов
    words = list(filter(lambda word: word not in auxiliary_words_porter, words))  # Удаление частиц, предлогов и тп
    return words


async def tokenize_stem(text):
    return tokenize_stem_sync(text)


async def text_stem_sync(text):
    words = tokenize_stem_sync(text)
    return " ".join(words)


# *******************************************************************************************************
# Чистка текста с помощью метода Портера.                                                               *
# @param text - исходный текст.                                                                         *
# @return обработанный текст.                                                                           *
# *******************************************************************************************************
async def text_stem(text):
    words = await tokenize_stem(text)
    return " ".join(words)


async def stemming_sentences(text):
    text = re.sub(r"\n", ".", text)
    text = re.sub(r'\.+', ".", text)
    text = re.sub(r'(?<=[.,])(?=[^\s])', r' ', text)
    sents = sent_tokenize(text)
    sents = [await text_stem(sent) for sent in sents]
    sents = list(filter(lambda sent: len(sent) > 3, sents))  # Удаление всего, что короче 2 символов
    sents = list(filter(lambda sent: sent.count(" ") > 2, sents))  # Удаление всего, что короче 2 символов
    return sents


async def stemming_sentences_par(par):
    sents = await stemming_sentences(par.text)
    return [CDTOParagraph(n=par.n, text=sent) for sent in sents]


async def stemming_sentences_indexed(dto_text: CDTOTextIndexed):
    dto_text.pars = [await stemming_sentences_par(par) for par in dto_text.pars]
    dto_text.pars = list(filter(lambda par: len(par) > 0, dto_text.pars))
    dto_text.pars = [par for arr in dto_text.pars for par in arr]
    return dto_text


# *******************************************************************************************************
# Чистка текста с помощью метода лемматизации.                                                          *
# @param text - исходный текст.                                                                         *
# @return обработанный текст.                                                                           *
# *******************************************************************************************************
async def text_lemmatization(text):
    words = await clear_text(text)
    words = list(map(lambda word: morph.parse(word)[0].normal_form, words))  # Лемматизация текста
    words = list(filter(lambda word: len(word) > 3, words))  # Удаление всего, что короче 3 символов

    return " ".join(words)
