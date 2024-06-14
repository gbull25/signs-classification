import glob
import logging
import random

import aiofiles
import aioredis
import emoji
import numpy as np
import requests
from aiocsv import AsyncReader, AsyncWriter
from aiogram import F, Router, types
from aiogram.filters.command import Command
from aiogram.types import FSInputFile, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder
from aiogram.utils.media_group import MediaGroupBuilder
from requests.exceptions import ConnectionError

logging.basicConfig(level=logging.INFO, format='%(asctime)s: [%(levelname)s] %(message)s')

router = Router()


# Хэндлер на команду /start
@router.message(Command("start"))
async def cmd_start(message: types.Message):
    redis = aioredis.from_url("redis://redis:5370")

    user_full_name = message.from_user.full_name
    user_id = message.from_user.id
    await message.answer(f'Привет, {user_full_name}\!\nЭтот бот умеет '
                         f'предсказывать класс немецких дорожных знаков\.\n'
                         f'Загрузи картинку со знаком или даже несколько, '
                         f'и бот попробует угадать, какой класс знака на них изображен\.')

    # Set default model for new user
    model = await redis.get(user_id)
    if model == None:
        await redis.set(user_id, "cnn")

    builder = ReplyKeyboardBuilder()

    bt1 = types.KeyboardButton(text="Информация")
    bt2 = types.KeyboardButton(text="Получить картинку")
    bt3 = types.KeyboardButton(text="Получить альбом")
    bt4 = types.KeyboardButton(text="Получить видео")
    bt5 = types.KeyboardButton(text="Оценить бота")
    bt6 = types.KeyboardButton(text="Текущий рейтинг")

    builder.row(bt1)
    builder.row(bt2, bt3, bt4)
    builder.row(bt5, bt6)

    await message.answer("Что бы вы хотели узнать?",
                         reply_markup=builder.as_markup(resize_keyboard=True))


# Хэндлер на команду информация
@router.message(F.text.lower() == "информация")
@router.message(Command('help'))
async def info(message: types.Message):
    text = ("Этот телеграм\-бот является частью студенческого "
            "[проекта\.](https://github.com/gbull25/signs-classification)\n"
            "Бот детектирует и предсказывает класс российских дорожных знаков\.")
    await message.answer(text)
    text = ("В настоящий момент в боте реализована модель детекции YOLO "
            "и модель классификации CNN\. Они работают последовательно "
            "для максимальной точности предсказания\.")
    await message.answer(text)
    text = ("Чтобы получить тестовое фото, набор фото или видео, нажмите на "
            "соответствующие кнопки в меню\.\nКонечно, вы также можете "
            "использовать собственные фото и видео\. Рекомендуем отправлять фотографии "
            "со сжатием\.")
    await message.answer(text)
    text = ("Если вы пользуетесь ботом с мобильного устройства, "
            "рекомендуется пересылать боту сообщения с полученными тестовыми файлами, "
            "без предварительного сохранения на внутреннюю память телефона\.")
    await message.answer(text)
    text = ("Если вы хотите оценить бота, нажмите на кнопку "
            "'Оценить бота'\.\nУчтите, что "
            "каждый пользователь может оставить только одну оценку, "
            "но ее можно менять, если вы передумаете\.")
    await message.answer(text)
    text = ("Чтобы получить теущий рейтинг бота, нажмите на кнопку "
            "'Текущий рейтинг'\.\nСпасибо и хорошего вам дня\!")
    await message.answer(text)


# Хэндлер на команду получить картинку
@router.message(F.text.lower() == "получить картинку")
@router.message(Command('image'))
async def upload_photo(message: Message):
    path = "sample_images/**"
    filename = random.choice(glob.glob(path))
    # Отправка файла из файловой системы
    image_from_pc = FSInputFile(path=filename)
    await message.answer_photo(
        image_from_pc,
        caption="Прошу, ваша тестовая картинка\!"
    )

# Хэндлер на команду получить видео
@router.message(F.text.lower() == "получить видео")
@router.message(Command('image'))
async def upload_video(message: Message):
    path = "sample_videos/**"
    filename = random.choice(glob.glob(path))
    # Отправка файла из файловой системы
    image_from_pc = FSInputFile(path=filename)
    await message.answer_video(
        image_from_pc,
        caption="Прошу, ваше тестовое видео\!"
    )

# Хэндлер на команду получить альбом
@router.message(F.text.lower() == "получить альбом")
@router.message(Command('images'))
async def upload_photos(message: Message):
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="1",
        callback_data="photos_1")
    )
    builder.add(types.InlineKeyboardButton(
        text="2",
        callback_data="photos_2")
    )
    builder.add(types.InlineKeyboardButton(
        text="3",
        callback_data="photos_3")
    )
    builder.add(types.InlineKeyboardButton(
        text="4",
        callback_data="photos_4")
    )
    builder.add(types.InlineKeyboardButton(
        text="5",
        callback_data="photos_5")
    )
    await message.answer(
        "Сколько картинок Вам отправить?",
        reply_markup=builder.as_markup()
    )


# Коллбэк на команду отправить альбом
@router.callback_query(F.data.startswith("photos_"))
async def send_album(callback: types.CallbackQuery):
    num_photos = int(callback.data.split("_")[1])
    # path = "services/tg_bot/sample_images/**"
    photos_paths = glob.glob("sample_images/**")

    album_builder = MediaGroupBuilder(
        caption=f"Прошу, ваши {num_photos} тестовых картинок\!"
    )

    for i in range(num_photos):
        random_photo = random.choice(photos_paths)
        album_builder.add(
            type="photo",
            media=FSInputFile(random_photo)
        )
        photos_paths.remove(random_photo)

    await callback.message.answer_media_group(
        media=album_builder.build()
    )


# Хэндлер на команду установить модель.
#@router.message(F.text.lower() == "выбрать модель")
#@router.message(Command('set_model'))
#async def set_model(message: types.Message):
#    builder = InlineKeyboardBuilder()
#    builder.add(types.InlineKeyboardButton(
#        text="CNN",
#        callback_data="model_cnn")
#    )
#    builder.add(types.InlineKeyboardButton(
#        text="SVC_SIFT",
#        callback_data="model_sift")
#    )
#    builder.add(types.InlineKeyboardButton(
#        text="SVC_HOG",
#        callback_data="model_hog")
#    )
#
#    await message.answer(
#        "Пожалуйста, выберите модель\.",
#        reply_markup=builder.as_markup()
#    )


# Калбэк на команду установить модель
#@router.callback_query(F.data.startswith("model_"))
#async def set_model_callback(callback: types.CallbackQuery):
#    redis = await aioredis.from_url("redis://redis:5370")
#    user_id = callback.from_user.id
#    model = callback.data.split('_')[1]
#    await redis.set(user_id, model)
#    await callback.answer(
#        f"Выбрана модель {model}\!",
#        show_alert=True
#    )


# Хэндлер на команду оценить бота
@router.message(F.text.lower() == "оценить бота")
@router.message(Command('rating'))
async def cmd_rating(message: types.Message):
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="1",
        callback_data="rating_1")
    )
    builder.add(types.InlineKeyboardButton(
        text="2",
        callback_data="rating_2")
    )
    builder.add(types.InlineKeyboardButton(
        text="3",
        callback_data="rating_3")
    )
    builder.add(types.InlineKeyboardButton(
        text="4",
        callback_data="rating_4")
    )
    builder.add(types.InlineKeyboardButton(
        text="5",
        callback_data="rating_5")
    )
    await message.answer(
        "Пожалуйста, оцените работу бота:",
        reply_markup=builder.as_markup()
    )
