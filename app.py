from flask import Flask, Blueprint
from flask_restplus import Api
from views.controller_string_clearing import ns_text_tools
from views.controller_lifecycle import ns_lifecycle

app = Flask(__name__)

api = Api(
    app,
    version='1.0',
    title='Text tools',
    description='The microservice for text lemmatization and stemming.'
)

# Подключаем все пространства имён.
api.add_namespace(ns_text_tools, path="/text")
api.add_namespace(ns_lifecycle, path="/lifecycle")