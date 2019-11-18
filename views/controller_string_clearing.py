from flask import request
from flask_restplus import Resource, Namespace, fields
from services.CServiceTextTools import text_processing_lemmatization, text_processing_porter
# *******************************************************************************************************
# Контроллер содержит обработку запросов на текстовые операции.                                         *
# @author Селетков И.П. 2019 1118.                                                                      *
# *******************************************************************************************************

ns_text_tools = Namespace('text tools', description='Text clearing tools')


# *******************************************************************************************************
# Выполнение лемматизации.                                                                              *
# *******************************************************************************************************
@ns_text_tools.route('/lemmatization')
@ns_text_tools.response(404, 'not found.')
class CTextLemmatization(Resource):
    def post(self):
        text = request.data.decode("utf-8")
        return text_processing_lemmatization(text)


# *******************************************************************************************************
# Выполнение стемминга.                                                                                 *
# *******************************************************************************************************
@ns_text_tools.route('/stem')
@ns_text_tools.response(404, 'not found.')
class CTextStemming(Resource):
    def post(self):
        text = request.data.decode("utf-8")
        return text_processing_porter(text)
