import numpy as np
import emoji
import aiofiles
import random
import glob
from aiogram import Router, F, types
from aiogram.types import Message
from aiogram.filters.command import Command
from aiogram.types import Message, FSInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiocsv import AsyncReader, AsyncWriter

router = Router()

# Хэндлер на команду /start
@router.message(Command("start"))
async def cmd_start(message: types.Message):
    #user_id = message.from_user.id
    user_full_name = message.from_user.full_name
    await message.answer(f'Привет, {user_full_name}\!\nЭтот бот умеет '
                         f'предсказывать класс немецких дорожных знаков\.\n'
                         f'Загрузи картинку со знаком или даже несколько, '
                         f'и бот попробует угадать, какой класс знака на них изображен\.')
    kb = [
        [
            types.KeyboardButton(text="Информация"),
            types.KeyboardButton(text="Получить картинку"),
            types.KeyboardButton(text="Оценить бота"),
            types.KeyboardButton(text="Текущий рейтинг")
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=kb,
        resize_keyboard=True,
        input_field_placeholder="Меню"
    )
    await message.answer("Что бы вы хотели узнать?", reply_markup=keyboard)

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
@router.message(Command('images'))
async def upload_photo(message: Message):
    #file_ids = []
    path = "services/tg_bot/sample_images/**"
    filename = random.choice(glob.glob(path))
    # Отправка файла из файловой системы
    image_from_pc = FSInputFile(path=filename)
    result = await message.answer_photo(
        image_from_pc,
        caption="Прошу, ваша тестовая картинка готова\!"
    )
    #file_ids.append(result.photo[-1].file_id)
    #await message.answer("Отправленные файлы:\n"+"\n".join(file_ids))


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
@router.callback_query(F.data.startswith("rating_"))
async def send_random_value(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    rating_value = int(callback.data.split("_")[1])

    async with aiofiles.open('./rating.csv', 'a',  encoding="utf-8", newline="") as f:
        writer = AsyncWriter(f)
        await writer.writerow([user_id, rating_value])

    await callback.answer(
        text="Спасибо, что воспользовались ботом!",
        show_alert=True
    )


# Хэндлер на команду текущий рейтинг
@router.message(F.text.lower() == "текущий рейтинг")
async def info(message: types.Message):
    rating_list = []
    await message.reply(f"Считаю текущий рейтинг бота\.\.")
    async with aiofiles.open('./rating.csv', mode="r", encoding="utf-8", newline="") as f:
        async for row in AsyncReader(f):
            if row[1] != 'rating':
                rating_list.append(int(row[1]))
    scale = int(np.floor(np.mean(rating_list)))
    await message.reply(scale * emoji.emojize(":star:"))