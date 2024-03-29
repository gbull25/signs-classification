import glob
import random
import logging

import aiofiles
import aioredis
import emoji
import numpy as np
from aiocsv import AsyncReader, AsyncWriter
from aiogram import F, Router, types
from aiogram.filters.command import Command
from aiogram.types import FSInputFile, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder
from aiogram.utils.media_group import MediaGroupBuilder

import requests
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
    bt4 = types.KeyboardButton(text="Оценить бота")
    bt5 = types.KeyboardButton(text="Текущий рейтинг")
    bt6 = types.KeyboardButton(text="Выбрать модель")

    builder.row(bt1)
    builder.row(bt2, bt3)
    builder.row(bt4, bt5, bt6)

    await message.answer("Что бы вы хотели узнать?",
                         reply_markup=builder.as_markup(resize_keyboard=True))


# Хэндлер на команду информация
@router.message(F.text.lower() == "информация")
@router.message(Command('help'))
async def info(message: types.Message):
    text = ("Этот телеграм\-бот является частью студенческого "
            "[проекта\.](https://github.com/gbull25/signs-classification)\n"
            "Бот предсказывает класс немецких знаков по фотографиям, "
            "используя для этого две ML модели семейства SVM, обученные на "
            "SIFT и HOG признаках\.")
    await message.answer(text)
    text = ("Чтобы получить тестовую фотографию, нажмите на "
            "кнопку 'Получить картинку'\.\nОтправьте фотографию "
            "или несколько фотографий со сжатием, чтобы получить "
            "предсказание класса знака обеими моделями для каждой из них\.")
    await message.answer(text)
    text = ("Если вы пользуетесь ботом с мобильного устройства, "
            "рекомендуется пересылать боту сообщения с полученными тестовыми картинками, "
            "без предварительного сохранения фотографий на внутреннюю память телефона, "
            "во избежание дополнительного сжатия\.")
    await message.answer(text)
    text = ("Если вы хотите оценить бота, нажмите на кнопку "
            "'Оценить бота'\.\nУчтите, что на текущий момент "
            "каждый пользователь может оставлять неограниченное "
            "количество оценок, сделано это с целью бесстыдной накрутки рейтинга\. "
            "В будущем, конечно, это будет изменено\.")
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
        caption="Прошу, ваша тестовая картинка готова\!"
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
        caption=f"Прошу, ваши {num_photos} тестовых картинок готовы\!"
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
@router.message(F.text.lower() == "выбрать модель")
@router.message(Command('set_model'))
async def set_model(message: types.Message):
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="CNN",
        callback_data="model_cnn")
    )
    builder.add(types.InlineKeyboardButton(
        text="SVC_SIFT",
        callback_data="model_sift")
    )
    builder.add(types.InlineKeyboardButton(
        text="SVC_HOG",
        callback_data="model_hog")
    )

    await message.answer(
        "Пожалуйста, выберите модель\.",
        reply_markup=builder.as_markup()
    )


# Калбэк на команду установить модель
@router.callback_query(F.data.startswith("model_"))
async def set_model_callback(callback: types.CallbackQuery):
    redis = await aioredis.from_url("redis://redis:5370")
    user_id = callback.from_user.id
    model = callback.data.split('_')[1]
    await redis.set(user_id, model)
    await callback.answer(
        f"Выбрана модель {model}\!",
        show_alert=True
    )


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


# Коллбэк на команду оценить бота
#@router.callback_query(F.data.startswith("rating_"))
#async def get_rating(callback: types.CallbackQuery):
#    user_id = callback.from_user.id
#    rating_value = int(callback.data.split("_")[1])
#
#    async with aiofiles.open('handlers/rating.csv',
#                             'a',  encoding="utf-8", newline="") as f:
#        writer = AsyncWriter(f)
#        await writer.writerow([user_id, rating_value])
#
#    await callback.answer(
#        text="Спасибо, что воспользовались ботом!",
#        show_alert=True
#    )


# Хэндлер на команду текущий рейтинг
#@router.message(F.text.lower() == "текущий рейтинг")
#async def current_rating(message: types.Message):
#    rating_list = []
#    await message.reply("Считаю текущий рейтинг бота\.\.")
#    async with aiofiles.open('handlers/rating.csv',
#                             mode="r", encoding="utf-8", newline="") as f:
#        async for row in AsyncReader(f):
#            if row[1] != 'rating':
#                rating_list.append(int(row[1]))
#    scale = int(np.floor(np.mean(rating_list)))
#    await message.reply(scale * emoji.emojize(":star:"))

