from datetime import datetime
from io import BytesIO
from unittest import mock

import emoji
import numpy as np
import pytest
import pytest_asyncio
import requests
from aiogram import Bot
from aiogram.filters import Command
from aiogram.types.input_file import FSInputFile
from aiogram_tests import MockedBot
from aiogram_tests.handler import CallbackQueryHandler, MessageHandler
from aiogram_tests.types.dataset import CALLBACK_QUERY, MESSAGE, MESSAGE_WITH_PHOTO
from handlers import menu, predictions


async def mock_download(*_args, **_kwargs):
    """Mocked download, imitate picture downloaded from tg."""
    with open('./sample_images/01576.png', 'rb') as f:
        io = BytesIO(f.read())
    return io


@pytest_asyncio.fixture
async def current_rating():
    """Fixture to get current rating from db."""
    response = requests.get("http://sign_classifier:80/rating/current_rating").json()
    return int(response["data"][0])


@pytest.mark.asyncio
async def test_current_rating(current_rating):
    """Test current_rating handler.
    
    Must return the same rating as it's observed from db.
    """
    requester = MockedBot(request_handler=MessageHandler(predictions.current_rating,
                                                         auto_mock_success=True))
    calls = await requester.query(MESSAGE.as_object(text="текущий рейтинг"))
    answer_message = calls.send_message.fetchone().text

    assert answer_message == current_rating * emoji.emojize(":star:"), \
        "Invalid rating calculation"


@pytest.mark.asyncio
async def test_upload_photo():
    """
    Test upload photo handler.

    Must return photo and correct caption.
    """
    requester = MockedBot(request_handler=MessageHandler(menu.upload_photo,
                                                         auto_mock_success=True))
    calls = await requester.query(MESSAGE.as_object(text="получить картинку"))
    answer_message = calls.send_photo.fetchone()

    assert hasattr(answer_message, "photo"), "Received reply has no attribute 'photo'."
    assert hasattr(answer_message, "caption"), "Received reply has no attribute 'caption'."
    assert isinstance(answer_message.photo, FSInputFile), (f"Received reply's 'photo' "
                                                           f"attribute content of a wrong"
                                                           f" type: {type(answer_message.photo)}.")
    assert answer_message.caption == r"Прошу, ваша тестовая картинка готова\!", \
        "Received reply's 'caption' attribute  has invalid content."


# upload_photos handler test
@pytest.mark.asyncio
async def test_upload_photos():
    """
    Test upload photos handler.

    Must return inline keyboard and correct capture.
    """
    requester = MockedBot(request_handler=MessageHandler(menu.upload_photos, auto_mock_success=True))
    calls = await requester.query(MESSAGE.as_object(text="получить альбом"))
    answer_message = calls.send_message.fetchone()

    assert hasattr(answer_message, "reply_markup"), \
        "Received reply has no attribute 'reply markup'."
    assert answer_message.reply_markup.get("inline_keyboard"), \
        "Received reply has no inline keyboard."
    assert len(answer_message.reply_markup.get("inline_keyboard")[0]) == 5, \
        (f"Received reply wrong "
         f"number of keyboard buttons:"
         f" {len(answer_message.reply_markup.get('inline_keyboard')[0])} != 5.")
    assert answer_message.text == "Сколько картинок Вам отправить?", \
        "Received reply has invalid content."


# send_album callback test
@pytest.mark.asyncio
async def test_send_album():
    """
    Test send_album callback.

    Must return correct number of photos and correct caption.
    """
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
           "Received reply's 'media' attribute's elements of a wrong type."
    assert all([isinstance(x["media"], FSInputFile) for x in answer_message.media]), \
           "Received reply's 'media' attribute contains media of a wrong type."
    assert answer_message.media[0]["caption"] == r"Прошу, ваши %s тестовых картинок готовы\!" % n_photos, \
        "Received reply has wrong caption."


@pytest.mark.asyncio
async def test_cmd_rating():
    """
    Test cmd_rating handler.

    Must return inline keyboard and correct caption.
    """
    requester = MockedBot(request_handler=MessageHandler(menu.cmd_rating, auto_mock_success=True))
    calls = await requester.query(MESSAGE.as_object(text="оценить бота"))
    answer_message = calls.send_message.fetchone()

    assert hasattr(answer_message, "reply_markup"), \
        "Received reply has no attribute 'reply markup'."
    assert answer_message.reply_markup.get("inline_keyboard"), \
        "Received reply has no inline keyboard."
    assert len(answer_message.reply_markup.get("inline_keyboard")[0]) == 5, \
        (f"Received reply wrong number of keyboard buttons: "
         f"{len(answer_message.reply_markup.get('inline_keyboard')[0])} != 5.")
    assert answer_message.text == "Пожалуйста, оцените работу бота:", \
        "Received reply has invalid content."


@pytest.mark.asyncio
async def test_add_rating():
    """
    Test add_rating handler.

    Must return correct caption and write new rating record to the db.
    """
    requester = MockedBot(CallbackQueryHandler(predictions.add_rating))

    # Make test data to write rating
    mark = np.random.randint(1, 6)
    data = f'rating_{mark}'

    callback_query = CALLBACK_QUERY.as_object(data=data, message=MESSAGE.as_object())
    calls = await requester.query(callback_query)
    answer_message = calls.answer_callback_query.fetchone()

    try:
        response = requests.post("http://sign_classifier:80/rating/get_rating_by_id", params={"user_id": 12345678}).json()
        requests.post("http://sign_classifier:80/rating/delete_rating", params={"user_id": 12345678})
    except ConnectionError:
        assert False, "Кажется, в настоящее время сервис прилег :\( Попробуйте еще разок позже\!"

    record_ts = datetime.strptime(response["data"]["timestamp"], "%Y-%m-%dT%H:%M:%S.%f")
    age = record_ts - datetime.utcnow()

    assert age.seconds > 10, "New rating has not been received, (record is too old.)"
    assert answer_message.text == "Спасибо, что воспользовались ботом!", \
        "Recieved reply has invalid content."


@pytest.mark.asyncio
async def test_cmd_start():
    """
    Test cmd_start handler.

    Must return correct caption.
    """
    requester = MockedBot(MessageHandler(menu.cmd_start, Command(commands=["start"])))
    calls = await requester.query(MESSAGE.as_object(text="/start"))

    # First message
    user_full_name = "FirstName LastName"
    answer_message = calls.send_message.fetchall()[0].text
    true_text = (f'Привет, {user_full_name}\\!\nЭтот бот умеет '
                 f'предсказывать класс немецких дорожных знаков\\.\n'
                 f'Загрузи картинку со знаком или даже несколько, '
                 f'и бот попробует угадать, какой класс знака на них изображен\\.')
    assert answer_message == true_text, "Recieved reply has invalid content."

    # Second message
    answer_message = calls.send_message.fetchall()[1].text
    true_text = 'Что бы вы хотели узнать?'
    assert answer_message == true_text, "Recieved reply has invalid content."


@mock.patch.object(Bot, 'download', mock_download, create=True)
@pytest.mark.asyncio
async def test_predict_image():
    """
    Test predict_image handler.

    Must correctly predict the image (its always the same) and return correct caption. 
    """
    requester = MockedBot(request_handler=MessageHandler(predictions.predict_image))
    calls = await requester.query(MESSAGE_WITH_PHOTO.as_object())
    answer_message = calls.send_message.fetchone().text
    true_text = ("*CNN* считает, что этот знак 38 класса \\(_Keep right_\\)\.")

    assert answer_message == true_text, "Recieved reply has invalid content."


@pytest.mark.asyncio
async def test_info():
    """
    Test info handler.

    Must return correct caption.
    """
    requester = MockedBot(MessageHandler(menu.info, auto_mock_success=True))
    calls = await requester.query(MESSAGE.as_object(text="информация"))

    # First message
    answer_message = calls.send_message.fetchall()[0].text
    true_text = ("Этот телеграм\\-бот является частью студенческого "
                 "[проекта\\.](https://github.com/gbull25/signs-classification)\n"
                 "Бот предсказывает класс немецких знаков по фотографиям, "
                 "используя для этого две ML модели семейства SVM, обученные на "
                 "SIFT и HOG признаках\\.")
    assert answer_message == true_text, "Recieved reply has invalid content."

    # Second message
    answer_message = calls.send_message.fetchall()[1].text
    true_text = ("Чтобы получить тестовую фотографию, нажмите на "
                 "кнопку 'Получить картинку'\\.\nОтправьте фотографию "
                 "или несколько фотографий со сжатием, чтобы получить "
                 "предсказание класса знака обеими моделями для каждой из них\\.")
    assert answer_message == true_text, "Recieved reply has invalid content."

    # Third message
    answer_message = calls.send_message.fetchall()[2].text
    true_text = ("Если вы пользуетесь ботом с мобильного устройства, "
                 "рекомендуется пересылать боту сообщения с полученными тестовыми картинками, "
                 "без предварительного сохранения фотографий на внутреннюю память телефона, "
                 "во избежание дополнительного сжатия\\.")
    assert answer_message == true_text, "Recieved reply has invalid content."

    # Fourth message
    answer_message = calls.send_message.fetchall()[3].text
    true_text = ("Если вы хотите оценить бота, нажмите на кнопку "
                 "'Оценить бота'\\.\nУчтите, что на текущий момент "
                 "каждый пользователь может оставлять неограниченное "
                 "количество оценок, сделано это с целью бесстыдной накрутки рейтинга\\. "
                 "В будущем, конечно, это будет изменено\\.")
    assert answer_message == true_text, "Recieved reply has invalid content."

    # Fifth message
    answer_message = calls.send_message.fetchall()[4].text
    true_text = ("Чтобы получить теущий рейтинг бота, нажмите на кнопку "
                 "'Текущий рейтинг'\\.\nСпасибо и хорошего вам дня\\!")
    assert answer_message == true_text, "Recieved reply has invalid content."
