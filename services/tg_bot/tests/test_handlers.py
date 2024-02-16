import pathlib

import emoji
import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
import aiofiles
from aiocsv import AsyncReader, AsyncWriter
from aiogram.filters import Command
from aiogram.types.input_file import FSInputFile
from aiogram_tests import MockedBot
from aiogram_tests.handler import CallbackQueryHandler, MessageHandler
from aiogram_tests.types.dataset import CALLBACK_QUERY, MESSAGE

from services.tg_bot.handlers import menu

RATING_FILE_PATH = pathlib.Path("services/tg_bot/handlers/rating.csv")
# RUN python -m pytest ./services/tg_bot


@pytest_asyncio.fixture
async def current_rating():
    rating_df = pd.read_csv(RATING_FILE_PATH)
    return int(np.floor(rating_df['rating'].mean()))


@pytest.mark.asyncio
async def test_current_rating(current_rating):
    requester = MockedBot(request_handler=MessageHandler(menu.current_rating, auto_mock_success=True))
    calls = await requester.query(MESSAGE.as_object(text="текущий рейтинг"))
    answer_message = calls.send_message.fetchone().text

    assert answer_message == current_rating * emoji.emojize(":star:"), "Invalid rating calculation"


@pytest.mark.asyncio
async def test_upload_photo():
    requester = MockedBot(request_handler=MessageHandler(menu.upload_photo, auto_mock_success=True))
    calls = await requester.query(MESSAGE.as_object(text="получить картинку"))
    answer_message = calls.send_photo.fetchone()

    assert hasattr(answer_message, "photo"), "Received reply has no attribute 'photo'."
    assert hasattr(answer_message, "caption"), "Received reply has no attribute 'caption'."
    assert isinstance(answer_message.photo, FSInputFile), (f"Received reply's 'photo' "
                                                           f"attribute content of a wrong"
                                                           f" type: {type(answer_message.photo)}.")
    assert answer_message.caption == "Прошу, ваша тестовая картинка готова\!", \
        "Received reply's 'caption' attribute  has invalid content."


# upload_photos handler test
@pytest.mark.asyncio
async def test_upload_photos_handler():
    requester = MockedBot(request_handler=MessageHandler(menu.upload_photos, auto_mock_success=True))
    calls = await requester.query(MESSAGE.as_object(text="получить альбом"))
    answer_message = calls.send_message.fetchone()

    assert hasattr(answer_message, "reply_markup"), "Received reply has no attribute 'reply markup'."
    assert answer_message.reply_markup.get("inline_keyboard"), "Received reply has no inline keyboard."
    assert len(answer_message.reply_markup.get("inline_keyboard")[0]) == 5, \
        (f"Received reply wrong "
         f"number of keyboard buttons:"
         f" {len(answer_message.reply_markup.get('inline_keyboard')[0])} != 5.")
    assert answer_message.text == "Сколько картинок Вам отправить?", "Received reply has invalid content."


# send_album callback test
@pytest.mark.asyncio
async def test_send_album_callback():
    requester = MockedBot(CallbackQueryHandler(menu.send_album))

    # Make test data to request n_photos in album
    n_photos = np.random.randint(1, 6)
    data = f'photos_{n_photos}'

    callback_query = CALLBACK_QUERY.as_object(data=data, message=MESSAGE.as_object())
    calls = await requester.query(callback_query)
    answer_message = calls.send_media_group.fetchone()

    assert hasattr(answer_message, "media"), "Recieved reply has no attribute 'media'."
    assert isinstance(answer_message.media, list), \
        (f"Received reply's 'media' attribute"
         f" content of a wrong type: {type(answer_message.media)}.")
    assert len(answer_message.media) == n_photos, \
        (f"Received reply's 'media' attribute content list of a wrong "
         f"length: {len(answer_message.media)} != {n_photos}.")
    assert all([isinstance(x, dict) for x in answer_message.media]), \
        f"Received reply's 'media' attribute's elements of a wrong type."
    assert all([isinstance(x["media"], FSInputFile) for x in answer_message.media]), \
        f"Received reply's 'media' attribute contains media of a wrong type."
    assert answer_message.media[0]["caption"] == f"Прошу, ваши {n_photos} тестовых картинок готовы\\!", \
        "Received reply has wrong caption."


@pytest.mark.asyncio
async def test_cmd_rating():
    requester = MockedBot(request_handler=MessageHandler(menu.cmd_rating, auto_mock_success=True))
    calls = await requester.query(MESSAGE.as_object(text="оценить бота"))
    answer_message = calls.send_message.fetchone()

    assert hasattr(answer_message, "reply_markup"), "Received reply has no attribute 'reply markup'."
    assert answer_message.reply_markup.get("inline_keyboard"), "Received reply has no inline keyboard."
    assert len(answer_message.reply_markup.get("inline_keyboard")[0]) == 5, \
        (f"Received reply wrong number of keyboard buttons: "
         f"{len(answer_message.reply_markup.get('inline_keyboard')[0])} != 5.")
    assert answer_message.text == "Пожалуйста, оцените работу бота:", "Received reply has invalid content."


@pytest.fixture
async def request_rating_1():
    rating_df = pd.read_csv(RATING_FILE_PATH)
    return rating_df['rating'].mean()


@pytest.fixture
async def request_rating_2():
    rating_df = pd.read_csv(RATING_FILE_PATH)
    return rating_df['rating'].mean()


@pytest.mark.asyncio
async def test_get_rating(request_rating_1, request_rating_2):

    # measure state of rating.csv before calling bot
    before_bot = request_rating_1

    requester = MockedBot(CallbackQueryHandler(menu.get_rating))

    # Make test data to write rating
    mark = np.random.randint(1, 6)
    data = f'rating_{mark}'

    callback_query = CALLBACK_QUERY.as_object(data=data, message=MESSAGE.as_object())
    calls = await requester.query(callback_query)
    answer_message = calls.answer_callback_query.fetchone()

    # measure state of rating.csv after calling bot
    after_bot = request_rating_2

    lines = []
    async with aiofiles.open('services/tg_bot/handlers/rating.csv', 
                             mode="r", encoding="utf-8", newline="") as f:
        async for row in AsyncReader(f):
            if row[1] != 'rating':
                lines.append(row)
        lines.pop()

    async with aiofiles.open('services/tg_bot/handlers/rating.csv', 
                             mode="w", encoding="utf-8", newline="") as f:
        writer = AsyncWriter(f)
        await writer.writerow(['user_id', 'rating'])
        for row in lines:
            await writer.writerow(row)

    assert before_bot != after_bot, "New rating has not been received"
    assert answer_message.text == "Спасибо, что воспользовались ботом!", "Recieved reply has invalid content."


@pytest.mark.asyncio
async def test_cmd_start():
    requester = MockedBot(MessageHandler(menu.cmd_start, Command(commands=["start"])))
    calls = await requester.query(MESSAGE.as_object(text="/start"))

    # First message
    user_full_name = "FirstName LastName"
    answer_message = calls.send_message.fetchall()[0].text
    true_text = (f'Привет, {user_full_name}\!\nЭтот бот умеет '
                f'предсказывать класс немецких дорожных знаков\.\n'
                f'Загрузи картинку со знаком или даже несколько, '
                f'и бот попробует угадать, какой класс знака на них изображен\.')
    assert answer_message == true_text, "Recieved reply has invalid content."

    # Second message
    answer_message = calls.send_message.fetchall()[1].text
    true_text = 'Что бы вы хотели узнать?'
    assert answer_message == true_text, "Recieved reply has invalid content."


@pytest.mark.asyncio
async def test_info():
    requester = MockedBot(MessageHandler(menu.info, auto_mock_success=True))
    calls = await requester.query(MESSAGE.as_object(text="информация"))

    # First message
    answer_message = calls.send_message.fetchall()[0].text
    true_text = ("Этот телеграм\-бот является частью студенческого "
            "[проекта\.](https://github.com/gbull25/signs-classification)\n"
            "Бот предсказывает класс немецких знаков по фотографиям, "
            "используя для этого две ML модели семейства SVM, обученные на "
            "SIFT и HOG признаках\.")
    assert answer_message == true_text, "Recieved reply has invalid content."

    # Second message
    answer_message = calls.send_message.fetchall()[1].text
    true_text = ("Чтобы получить тестовую фотографию, нажмите на "
            "кнопку 'Получить картинку'\.\nОтправьте фотографию "
            "или несколько фотографий со сжатием, чтобы получить "
            "предсказание класса знака обеими моделями для каждой из них\.")
    assert answer_message == true_text, "Recieved reply has invalid content."

    # Third message
    answer_message = calls.send_message.fetchall()[2].text
    true_text = ("Если вы пользуетесь ботом с мобильного устройства, "
            "рекомендуется пересылать боту сообщения с полученными тестовыми картинками, "
            "без предварительного сохранения фотографий на внутреннюю память телефона, "
            "во избежание дополнительного сжатия\.")
    assert answer_message == true_text, "Recieved reply has invalid content."

    # Fourth message
    answer_message = calls.send_message.fetchall()[3].text
    true_text = ("Если вы хотите оценить бота, нажмите на кнопку "
            "'Оценить бота'\.\nУчтите, что на текущий момент "
            "каждый пользователь может оставлять неограниченное "
            "количество оценок, сделано это с целью бесстыдной накрутки рейтинга\. "
            "В будущем, конечно, это будет изменено\.")
    assert answer_message == true_text, "Recieved reply has invalid content."

    # Fifth message
    answer_message = calls.send_message.fetchall()[4].text
    true_text = ("Чтобы получить теущий рейтинг бота, нажмите на кнопку "
            "'Текущий рейтинг'\.\nСпасибо и хорошего вам дня\!")
    assert answer_message == true_text, "Recieved reply has invalid content."
