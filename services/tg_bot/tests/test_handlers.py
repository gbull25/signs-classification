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
    assert isinstance(answer_message.photo, FSInputFile), f"Recieved reply 'photo' content of a wrong type: {type(answer_message.photo)}."
    assert answer_message.caption == "Прошу, ваша тестовая картинка готова\!", "Recieved reply 'caption' has invalid content."


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
