import pathlib

import emoji
import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from aiogram.filters import Command
from aiogram.methods import AnswerCallbackQuery, SendMessage
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
    assert isinstance(answer_message.photo, FSInputFile), f"Recieved reply's 'photo' attribute content of a wrong type: {type(answer_message.photo)}."
    assert answer_message.caption == "Прошу, ваша тестовая картинка готова\!", "Recieved reply's 'caption' attribute  has invalid content."


# upload_photos handler test
@pytest.mark.asyncio
async def test_upload_photos_handler():
    requester = MockedBot(request_handler=MessageHandler(menu.upload_photos, auto_mock_success=True))
    calls = await requester.query(MESSAGE.as_object(text="получить альбом"))
    answer_message = calls.send_message.fetchone()

    assert hasattr(answer_message, "reply_markup"), "Recieved reply has no attribute 'reply markup'."
    assert answer_message.reply_markup.get("inline_keyboard"), "Recieved reply has no inline keyboard."
    assert len(answer_message.reply_markup.get("inline_keyboard")[0]) == 5, f"Recieved reply wrong number of keyboard buttons: {len(answer_message.reply_markup.get('inline_keyboard')[0])} != 5."
    assert answer_message.text == "Сколько картинок Вам отправить?", "Recieved reply has invalid content."


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
    assert isinstance(answer_message.media, list), f"Recieved reply's 'media' attribute content of a wrong type: {type(answer_message.media)}."
    assert len(answer_message.media) == n_photos, f"Recieved reply's 'media' attribute content list of a wrong length: {len(answer_message.media)} != {n_photos}."
    assert all([isinstance(x, dict) for x in answer_message.media]), f"Recieved reply's 'media' attribute's elements of a wrong type."
    assert all([isinstance(x["media"], FSInputFile) for x in answer_message.media]), f"Recieved reply's 'media' attribute contains media of a wrong type."
    assert answer_message.media[0]["caption"] == f"Прошу, ваши {n_photos} тестовых картинок готовы\\!", "Recieved reply has wrong caption."


@pytest.mark.asyncio
async def test_cmd_start():
    requester = MockedBot(MessageHandler(menu.cmd_start, Command(commands=["start"])))
    calls = await requester.query(MESSAGE.as_object(text="/start"))

    # Для первого сообщения
    user_full_name = "FirstName LastName"
    answer_message = calls.send_message.fetchall()[0].text
    true_text = (f'Привет, {user_full_name}\!\nЭтот бот умеет '
                f'предсказывать класс немецких дорожных знаков\.\n'
                f'Загрузи картинку со знаком или даже несколько, '
                f'и бот попробует угадать, какой класс знака на них изображен\.')
    assert answer_message == true_text, "Recieved reply has invalid content."

    # Для второго сообщения
    answer_message = calls.send_message.fetchall()[1].text
    true_text = 'Что бы вы хотели узнать?'
    assert answer_message == true_text, "Recieved reply has invalid content."
