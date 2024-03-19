import logging
from io import BytesIO

import aioredis
import numpy as np
import requests
import emoji
from aiogram import Bot, F, Router, types
from aiogram.types import InputMediaPhoto, Message
from requests.exceptions import ConnectionError

router = Router()
logging.basicConfig(level=logging.INFO, format='%(asctime)s: [%(levelname)s] %(message)s')


# Хэндлер на альбом фотографий
@router.message(F.media_group_id, F.content_type.in_({'photo'}))
async def handle_albums(message: Message, album: list[Message], bot: Bot):
    media_group = []
    pred_result = []

    redis = await aioredis.from_url("redis://redis:5370")
    user_id = message.from_user.id
    model = await redis.get("user_id")
    if model == None:
        model = "cnn"
        await redis.set(user_id, "cnn")

    # aiohttp WIP
    # request_data = aiohttp.FormData()

    for i, msg in enumerate(album):
        if msg.photo:

            file_id = msg.photo[-1].file_id
            media_group.append(InputMediaPhoto(media=file_id))

            io = BytesIO()
            await bot.download(msg.photo[-1], destination=io)
            im = io.getvalue()

            # aiohttp WIP
            # request_data.add_field('file',
            #    im,
            #    filename=f'file_{i}',
            #    content_type='image/png')

            try:

                response = requests.post(
                    "http://sign_classifier:80/classify_sign",
                    params={'model_name': model},
                    files={'file_img': im}
                ).json()
                logging.info(f"Received a response with prediciton: {response}")
                pred_result.append((response['sign_class'], response['sign_description']))

            except ConnectionError as ce:

                logging.error(f"Connection refused error: {ce}")
                await message.reply("Кажется, в настоящее время сервис прилег :\( Попробуйте еще разок позже\!")
                return

    # aiohttp WIP
    # async with aiohttp.ClientSession() as session:
    #     res = await session.post("http://sign_classifier:80/predict/signs_cnn", data=request_data)

    # await message.answer(res.status)

    # Возвращаем альбом для удобства чтения результатов классификации
    await message.answer_media_group(media_group)
    # Возвращаем предсказания
    i = 0
    for pred in pred_result:
        i += 1
        await message.reply(f'На фотографии номер {i}:\n'
                            f'*{model.upper()}* считает, что знак {pred[0]} класса \(_{pred[1]}_\)\.')


# Хэндлер на одну фотографию
@router.message(F.photo)
async def predict_image(message: Message, bot: Bot):
    io = BytesIO()
    io = await bot.download(message.photo[-1], destination=io)
    im = io.getvalue()

    redis = await aioredis.from_url("redis://redis:5370")
    user_id = message.from_user.id
    model = await redis.get("user_id")
    if model == None:
        model = "cnn"
        await redis.set(user_id, "cnn")   

    try:

        response = requests.post(
                    "http://sign_classifier:80/classify_sign",
                    params={'model_name': model},
                    files={'file_img': im}
        ).json()
        logging.info(f"Received a response with prediciton: {response}")

    except ConnectionError as ce:

        logging.error(f"Connection refused error: {ce}")
        await message.reply("Кажется, в настоящее время сервис прилег :\( Попробуйте еще разок позже\!")
        return

    await message.reply(f'*{model.upper()}* считает, что этот знак '
                        f'{response["sign_class"]} класса \(_{response["sign_description"]}_\)\.')


# Хэндлер на рейтинг
@router.callback_query(F.data.startswith("rating_"))
async def add_rating(callback: types.CallbackQuery):

    data = {"user_id": callback.from_user.id,
            "rating": int(callback.data.split("_")[1])}

    try:

        requests.post("http://sign_classifier:80/rating/add_rating", json = data)

    except ConnectionError as ce:

        logging.error(f"Connection refused error: {ce}")
        await message.reply("Кажется, в настоящее время сервис прилег :\( Попробуйте еще разок позже\!")
        return

    await callback.answer(
        text="Спасибо, что воспользовались ботом!",
        show_alert=True
    )


@router.message(F.text.lower() == "текущий рейтинг")
async def current_rating(message: types.Message):
    await message.reply("Считаю текущий рейтинг бота\.\.")

    try:

        scale = requests.get("http://sign_classifier:80/rating/current_rating").json()

    except ConnectionError as ce:

        logging.error(f"Connection refused error: {ce}")
        await message.reply("Кажется, в настоящее время сервис прилег :\( Попробуйте еще разок позже\!")
        return

    await message.reply(int(np.floor(scale["data"])) * emoji.emojize(":star:"))