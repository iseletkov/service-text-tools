FROM iseletkov/fastapi:buster-3.9-0.63.0
LABEL maintainer="iseletkov@gmail.com"

# Загрузка дополнительных ресурсов для nltk
RUN python -c "import nltk;nltk.download('punkt');nltk.download('stopwords')"
# Порт, на котором работает сервер
EXPOSE 8000
# Копирование папки с приложением в контейнер
COPY ./app /app
#Команда для запуска сервера.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

