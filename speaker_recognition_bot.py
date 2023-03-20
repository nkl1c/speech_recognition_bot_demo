import soundfile as sf
import yaml
import numpy as np
import telegram
from telegram.ext import *
import io
import librosa
from PIL import Image
from inference import (build_classifier_speechbrain, 
    build_label_decoder, build_name_decoder,
    main as recognize)



bot = telebot.TeleBot('6134902648:AAFDPQBp2ZoEXuhntTnAJgexVIBmlY_WqZo');

@bot.message_handler(commands=['start'])
def start(message):
  bot.send_message(message.chat.id,f"Привет, {message.from_user.first_name}!👋")
  markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width = 2)
  btn1 = types.KeyboardButton("LooksLike")
  btn2 = types.KeyboardButton('Speech->Text')
  btn3 = types.KeyboardButton('3D Model')
  btn4 = types.KeyboardButton('Skleyka')
  btn5 = types.KeyboardButton('PhotoDescriber')
  btn6 = types.KeyboardButton('CelebrityVoice')

  bot.send_message(message.from_user.id, "Спасибо за разработку (или просто готовый код):\nLooksLike - @sslatyshev\nSpeech->Text - @EnderPortman\n3D Model - @cloud_01_24\nSkleyka - @cloud_01_24\nPhotoDescriber - @cloud_01_24\nCelebrityVoice - @nkl_c & @elizzz13\nBot - @EnderPortman")

  markup.add(btn1, btn2, btn3, btn4, btn5, btn6)
  bot.send_message(message.from_user.id, "Выбери действие🔽", reply_markup=markup)


task_type = 0
@bot.message_handler(content_types=['text'])
def text(message):
  global task_type
  mess = message.text

  if mess == "LooksLike":
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
    btn = types.KeyboardButton("Хочу выбрать снова↩️")
    markup.add(btn)
    task_type = 1
    ans = "Жду твое фото😊📷"
    bot.send_message(message.from_user.id, ans, reply_markup=markup)
  
  elif mess == "Speech->Text":
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
    btn = types.KeyboardButton("Хочу выбрать снова↩️")
    markup.add(btn)
    task_type = 2
    ans = "Жду аудио🎧"
    bot.send_message(message.from_user.id, ans, reply_markup=markup)
    bot.send_message(message.from_user.id, "Гс не более 15 сек на русском\nФункция может сработать не сразу, так как это бесплатно :)")

  elif mess == "3D Model":
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
    btn = types.KeyboardButton("Хочу выбрать снова↩️")
    markup.add(btn)
    task_type = 3
    ans = "Жду картинку🌇"
    bot.send_message(message.from_user.id, ans, reply_markup=markup)

  elif mess == "Skleyka":
      markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
      btn = types.KeyboardButton("Хочу выбрать снова↩️")
      markup.add(btn)
      task_type = 4
      ans = "Жду два фото с разного ракурса🌇🌆"
      bot.send_message(message.from_user.id, ans, reply_markup=markup)
  
  elif mess == "PhotoDescriber":
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
    btn = types.KeyboardButton("Хочу выбрать снова↩️")
    markup.add(btn)
    task_type = 5
    ans = "Жду картинку зверя🐸 или чего угодно🎃"
    bot.send_message(message.from_user.id, ans, reply_markup=markup)

  elif mess == "CelebrityVoice":
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
    btn = types.KeyboardButton("Хочу выбрать снова↩️")
    markup.add(btn)
    task_type = 6
    ans = "Скажи мне что-нибудь🗣"
    bot.send_message(message.from_user.id, ans, reply_markup=markup)

  elif mess == "Хочу выбрать снова↩️" or mess == "Круто! Давай ещё🤩" or mess == "👌" or mess == "Давай!":
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    btn1 = types.KeyboardButton("LooksLike")
    btn2 = types.KeyboardButton('Speech->Text')
    btn3 = types.KeyboardButton('3D Model')
    btn4 = types.KeyboardButton('Skleyka')
    btn5 = types.KeyboardButton('PhotoDescriber')
    btn6 = types.KeyboardButton('CelebrityVoice')

    markup.add(btn1, btn2, btn3, btn4, btn5, btn6)
    ans = "Выбери действие🔽"
    bot.send_message(message.from_user.id, ans, reply_markup=markup) 

  else:
    print(task_type)
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
    btn = types.KeyboardButton("Хочу выбрать снова↩️")
    task_type = 0
    markup.add(btn)
    bot.send_message(message.from_user.id, text = "🐈", reply_markup=markup)    

count_files = 0
@bot.message_handler(content_types=['photo'])
def photo(message):
  global task_type
  global count_files
  if task_type in [1, 3, 4, 5]:
    count_files += 1

    if count_files == 2 and task_type == 4:
      photo_id = message.photo[-1].file_id
      photo_file = bot.get_file(photo_id)
      photo_bytes = bot.download_file(photo_file.file_path)
      with open("im2.jpg", 'wb') as new_file:
            new_file.write(photo_bytes)

      Skleyka("im1.jpg", "im2.jpg")
      photo = open('result.jpg', 'rb')
      bot.send_photo(message.chat.id, photo)
      
      ans = 'И как?🧐'
      markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
      btn = types.KeyboardButton("Круто! Давай ещё🤩")
      markup.add(btn)
      bot.send_message(message.from_user.id, ans, reply_markup=markup)
      task_type = 0
      count_files = 0

    elif count_files == 1:
      photo_id = message.photo[-1].file_id
      photo_file = bot.get_file(photo_id) 
      photo_bytes = bot.download_file(photo_file.file_path)
      with open("im1.jpg", 'wb') as new_file:
            new_file.write(photo_bytes)

      if task_type == 1:
        bot.send_message(message.from_user.id, 'Я тебя узнал!\nМне нужно 8 минут, чтобы найти фото...')
        bot.send_message(message.from_user.id, '👀')
        im1 = LooksLike("im1.jpg")
        bot.send_photo(message.chat.id, im1)
        ans = 'И как?🧐'
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
        btn = types.KeyboardButton("Круто! Давай ещё🤩")
        markup.add(btn)
        bot.send_message(message.from_user.id, ans, reply_markup=markup)
        task_type = 0
        count_files = 0

      elif task_type == 3:
        DDD_model("im1.jpg")
        doc = open('/content/model.stl')
        bot.send_document(message.chat.id, doc)
        ans = 'И как?🧐'
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
        btn = types.KeyboardButton("Круто! Давай ещё🤩")
        markup.add(btn)
        bot.send_message(message.from_user.id, ans, reply_markup=markup)
        task_type = 0
        count_files = 0
      
      elif task_type == 5:
        text = things_classification("im1.jpg")
        bot.send_message(message.from_user.id, 'Вероятности c описанием готовы!🥳')
        bot.send_message(message.from_user.id, text)
        ans = 'И как?🧐'
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
        btn = types.KeyboardButton("Круто! Давай ещё🤩")
        markup.add(btn)
        bot.send_message(message.from_user.id, ans, reply_markup=markup)
        task_type = 0
        count_files = 0      

  else:
    task_type = 0
    count_files = 0
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
    btn = types.KeyboardButton("Хочу выбрать снова↩️")
    markup.add(btn)
    bot.send_message(message.from_user.id, text = "🐈", reply_markup=markup)

@bot.message_handler(content_types=['voice'])
def voice_rec(message):
  global task_type
  if task_type in [2, 6]:
    if task_type == 2:
      bot.send_message(message.from_user.id, "Какой красивый голос🥰")
      filename = str(uuid.uuid4())
      file_name_full=filename+".ogg"
      file_name_full_converted=filename+".wav"
      file_info = bot.get_file(message.voice.file_id)
      downloaded_file = bot.download_file(file_info.file_path)
      with open(file_name_full, 'wb') as new_file:
          new_file.write(downloaded_file)
      os.system("ffmpeg -i "+file_name_full+"  "+file_name_full_converted)
      text=speech(file_name_full_converted)

      bot.send_message(message.from_user.id, text = text)
      ans = 'Хочешь ещё?🤔'
      markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
      btn = types.KeyboardButton("Давай!")
      markup.add(btn)
      bot.send_message(message.from_user.id, ans, reply_markup=markup)
      task_type = 0 

    elif task_type == 6:
      bot.send_message(message.from_user.id, "Приятный голос🤤")
      bot.send_message(message.from_user.id, "Подожди немного..")
      filename = str(uuid.uuid4())
      file_name_full=filename+".ogg"
      file_name_full_converted=filename+".wav"
      file_info = bot.get_file(message.voice.file_id)
      downloaded_file = bot.download_file(file_info.file_path)
      with open(file_name_full, 'wb') as new_file:
          new_file.write(downloaded_file)
      os.system("ffmpeg -i "+file_name_full+"  "+file_name_full_converted)

      CelebrityVoice(file_name_full_converted)

      bot.send_message(message.from_user.id, "У тебя такой же голос, как у....")
      photo = open('frame_0.jpg', 'rb')
      bot.send_photo(message.from_user.id, photo)

      ans = 'Хочешь ещё?🤔'
      markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
      btn = types.KeyboardButton("Давай!")
      markup.add(btn)
      bot.send_message(message.from_user.id, ans, reply_markup=markup)
      task_type = 0 

  else:
    task_type = 0
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
    btn = types.KeyboardButton("Хочу выбрать снова↩️")
    markup.add(btn)
    bot.send_message(message.from_user.id, text = "🐈", reply_markup=markup)  
  

bot.polling(none_stop=True, interval=0)
