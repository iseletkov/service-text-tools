FROM python:latest
LABEL maintainer="iseletkov@gmail.com"
# Установка модулей python
RUN /usr/local/bin/pip install fastapi uvicorn pymorphy2 nltk pydantic sklearn
# Загрузка дополнительных ресурсов для nltk
RUN python -c "import nltk;nltk.download('punkt');nltk.download('stopwords')"
# Порт, на котором работает сервер
EXPOSE 8000
# Копирование папки с приложением в контейнер
COPY ./app /app
#Команда для запуска сервера.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

