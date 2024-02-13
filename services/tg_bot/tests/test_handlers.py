import pathlib

import emoji
import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from aiogram.filters import Command
from aiogram.methods import AnswerCallbackQuery, SendMessage
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
    true_rating = current_rating
    assert answer_message == true_rating * emoji.emojize(":star:")