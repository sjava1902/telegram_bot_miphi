# -*- coding: utf-8 -*-
import telebot
from telebot import types
import nltk
from nltk.metrics.distance import edit_distance
import dateparser
import requests
import whisper
import subprocess
import os, sys
from pathlib import Path
import datetime
from datetime import date
from pyaspeller import YandexSpeller
import pandas as pd
import joblib
import lightgbm
from dateutil.relativedelta import relativedelta
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from difflib import SequenceMatcher
from natasha import (
    Segmenter,
    MorphVocab,   
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,  
    PER,
    NamesExtractor,
    DatesExtractor,
    MoneyExtractor,
    AddrExtractor,
    Doc
)
from pymystem3 import Mystem

nltk.download('stopwords')
nltk.download('punkt')

current_date = date.today()
initial_date = date(2024, 4, 3)
token='6463803088:AAH9iVZAv71ScKwERcscNCTyY5-4wkqZ1Ew'
bot=telebot.TeleBot(token)
time_words = ['вчера', 'сегодня', 'завтра', 'послезавтра', 'позавчера', 'через', 'после', 'назад', 'следующ', 'прошл', 'недел', 'прошедш'
              'январ', 'феврал', 'март', 'апрел', 'май', 'мая', 'июн', 'июл', 'август', 'сентябр', 'октябр', 'ноябр', 'декабр',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'понедельник', 'вторник', 'сред', 'четверг', 'пятниц', 'суббот', 'воскресенье',
              'один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять', '.', ':', ';', '-', '/', '\\', '|',
              'десять', 'двадцать', 'тридцать', 'сорок', 'пятьдесят', 'шестьдесят', 'семьдесят', 'восемьдесят', 'девяносто', 'сто',
              'дней', 'дня', 'дню', 'месяц', 'год', 'лет',
              
              'yesterday', 'today', 'tomorrow', 'the day after tomorrow', 'the day before yesterday', 'through', 'field', 'back', 'next', 'passed', 'weeks', 'past'
              'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
              'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', '.', ':', ',', ';', '-', '/', '\\', '|',
              'ten', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred']
delta_time_words = ['вчера', 'сегодня', 'завтра', 'послезавтра', 'позавчера', 'через', 'назад']

load_dir = 'models'
df = pd.read_excel('data_final.xlsx')
# Получаем список станций
stations = df.columns[1:]  # Первый столбец содержит даты, поэтому начинаем с индекса 1
print("before")
for station in stations:
    station = station.split(' ')[0]
print("after")
stations_preprocessed = [x.lower() for x in stations]
print(stations_preprocessed)

station_models={}
loaded_models = {}

# Перебираем каждую станцию
for i, station in enumerate(stations):
    if station == 'Date':
        continue
    # Сохранение модели
    # Сохранение модели в папку
    filename = os.path.join(load_dir, f"{station}_model.joblib")
    # Добавление модели в словарь
    station_models[station] = filename
    # Загрузка моделей
    loaded_models[station] = joblib.load(filename)

print(stations)

def predict_station(sentence):
    max_similarity = 0
    predicted_station = None
    for word in sentence.split():
        for station in stations_preprocessed:
            station = station.split(' ')[0]
            print("!!!", station)
            similarity = SequenceMatcher(None, word.lower(), station.lower()).ratio()
            if similarity > max_similarity:
                max_similarity = similarity
                predicted_station = station
    return predicted_station

def find_count(days, station):
    max_date = df['Date'].max()  
    start_date = max_date - pd.Timedelta(days=days)  
    start_index = df[df['Date'] == start_date].index
    count = df.loc[start_index, station].iloc[0]
    return int(count)

def find_future(days, station):    
    # Выбор соответствующей модели для станции
    model = loaded_models[station]
    
    # Генерация даты в будущем
    date = df['Date'].max() + pd.Timedelta(days=days)
    
    # Создание DataFrame для прогнозирования
    dat = pd.DataFrame({'Date': [date]})
    dat['day'] = dat['Date'].dt.day
    dat['weekday'] = (dat['Date'].dt.dayofweek < 5).astype(int)
    dat['weekend'] = (dat['Date'].dt.dayofweek >= 5).astype(int)
    dat['month'] = dat['Date'].dt.month
    
    # Прогнозирование и округление
    pred = model.predict(dat.iloc[:, 1:])
    prediction = int(round(pred[0]))
    
    return prediction, date

def remove_stop_words(text):
    stop_words = set(stopwords.words('russian'))
    word_tokens = word_tokenize(text)
    #print("word_tokens = ", word_tokens)
    # converts the words in word_tokens to lower case and then checks whether 
    #they are present in stop_words or not
    #filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    #with no lower case conversion
    filtered_text = ""
    for w in word_tokens:
        if w not in stop_words:
            filtered_text += w + " "
    return filtered_text


segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
def lemmatization(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)

    res = ""
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        res += token.lemma + " "
    return res
 
print("LEMMATIZATION START")
for station in stations:
    station = lemmatization(station)
print("LEMMATIZATION DONE")
    

def preprocess(text):
    text = text.lower()
    text = remove_stop_words(text)
    #text = lemmatization(text)
    text = text.replace(',', '')
    text = text.replace('.', '')
    speller = YandexSpeller()
    text = speller.spelled(text)
    print("PREPROCESSED TEXT = ", text)
    return text

def get_time_substr(sentence, time_words):
    words = sentence.split(' ')
    time_res, other_res = [], []
    for word in words:
        isTimeWord = False
        for time_word in time_words:
            if time_word in word:
                time_res.append(word)
                isTimeWord = True
                break
        if not isTimeWord:
            other_res.append(word)
            
    res_str = ""
    for elem in time_res:
        res_str += elem + ' '
        
    other_str = ""
    for elem in other_res:
        other_str += elem + ' '
        
    return res_str, other_str

def get_word_by_min_distance(sentence):
    #sentence = sentence.split(' ')
    close_word = ""
    min_distance = 200
    for station in stations_preprocessed:
        distance = edit_distance(sentence, station)
        if(distance < min_distance):
            min_distance = distance
            close_word = station
    for station in stations:
        station_preprocessed = station.lower()
        if(close_word == station_preprocessed):
            close_word = station
    print("MIN DISTANCE = ", min_distance)
    if(min_distance >= min(len(close_word), len(sentence))):
        return ""
    return close_word
    

# Handle '/start' and '/help'
@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "Введите какую информацию о пассажиропотоке в какой день и на какой станции Вы хотите узнать!")
    
@bot.message_handler(commands=['help'])
def help(messsage):
    bot.reply_to(messsage, "Доступные команды: /start, /help")
    
@bot.message_handler(content_types=['photo'])
def photo(message):
    bot.send_message(message.chat.id, "Фото не поддерживается.")

# Handle all other messages with content_type 'text' (content_types defaults to ['text'])
@bot.message_handler(content_types=['text'])
def text(message):
    initial_date = date(2024, 4, 3)
    text = message.text
    text = preprocess(text)
    isRelativeDate = False
    time_substr, other_substr = get_time_substr(text, delta_time_words)
    if(time_substr):
        time_substr, other_substr = get_time_substr(text, time_words)
        isRelativeDate = True
    else:
        time_substr, other_substr = get_time_substr(text, time_words)
    time_substr = time_substr.strip(" ")
    other_substr = other_substr.strip(" ")
    print('TIME_SUBSTR = ', time_substr)
    print('OTHER_SUBSTR = ', other_substr)
    time = dateparser.parse(time_substr)
    print("TIME = ", time)
    stationIsCorrect = False
    timeIsCorrect = False
    
    curr_station = predict_station(other_substr)
    if curr_station:
        stationIsCorrect = True
        #bot.reply_to(message, "Введите корректную станцию")
    #curr_station = get_word_by_min_distance(curr_station)
    #print("CURR_STATION == ", curr_station)
    for s in stations:
        if curr_station in s.lower():
            curr_station = s
            break
            
    
    # for station in stations_preprocessed:
    #     if station in other_substr:
    #         stationIsCorrect = True
    #         #bot.reply_to(message, station)
    #         print("STATION BEFORE MID DISTANCE = ", station)
    #         station = get_word_by_min_distance(station)
    #         print("STATION AFTER MIN DISTANCE = ", station)
    #         break 
    # if not stationIsCorrect:
    #     station = get_word_by_min_distance(other_substr)
    #     if station:
    #         #bot.reply_to(message, station)
    #         stationIsCorrect = True
    #     else:
    #         bot.reply_to(message, "Введите корректную станцию")

    if(time):
        #bot.reply_to(message, time)
        timeIsCorrect = True
    else:
        bot.reply_to(message, "Введите корректное время")
    
    if stationIsCorrect and timeIsCorrect:
        count_days = 0
        if isRelativeDate:
            count_days = (current_date - time.date()).days
        else:
            count_days = (initial_date - time.date()).days
        print("COUNT = ", count_days)
        initial_date -= relativedelta(days=count_days)
        print("STATION TO MODEL = ", curr_station)
        if count_days >=0 :
            count = find_count(count_days, curr_station)
            output_message = f'На станции {curr_station} число пассажиров {count}. Дата {initial_date}'
        else:
            count, _ = find_future(count_days, curr_station)
            output_message = f'На станции {curr_station} прогнозируемое число пассажиров {count}. Дата {initial_date}'
        bot.send_message(message.chat.id, output_message)
        
@bot.message_handler(content_types=['voice'])
def audio(message):
    file_info = bot.get_file(message.voice.file_id)
    path = file_info.file_path # Вот тут-то и полный путь до файла (например: voice/file_2.oga)
    fname = os.path.basename(path) # Преобразуем путь в имя файла (например: file_2.oga)
    print(fname)
    doc = requests.get('https://api.telegram.org/file/bot{0}/{1}'.format(token, file_info.file_path)) # Получаем и сохраняем присланную голосвуху (Ага, админ может в любой момент отключить удаление айдио файлов и слушать все, что ты там говоришь. А представь, что такую бяку подселят в огромный чат и она будет просто логировать все сообщения [анонимность в телеграмме, ахахаха])
    with open(fname, 'wb') as f:
        f.write(doc.content) # вот именно тут и сохраняется сама аудио-мессага
    subprocess.run(['ffmpeg', '-i', fname, fname+'.wav'])# здесь используется страшное ПО ffmpeg, для конвертации .oga в .vaw
    whisper_model = whisper.load_model("small")
    result = whisper_model.transcribe(fname+'.wav')
    bot.send_message(message.from_user.id, format(result['text'])) # Отправляем пользователю, приславшему файл, его текст
    message.text = result['text']
    text(message)
    os.remove(fname)
    os.remove(fname+".wav")

if __name__ == '__main__':
    bot.infinity_polling()