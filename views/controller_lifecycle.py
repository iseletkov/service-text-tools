from flask import request
from flask_restplus import Resource, Namespace

# *******************************************************************************************************
# Контроллер содержит методы для управления жизненным циклом сервиса.                                   *
# *******************************************************************************************************

ns_lifecycle = Namespace('lifecycle', description='Lifecycle operations')


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
# *******************************************************************************************************
# Остановка сервера.                                                                                    *
# *******************************************************************************************************
@ns_lifecycle.route('/shutdown')
class CShutdown(Resource):
    def get(self):
        shutdown_server()
        return 'Server is shutting down...'
