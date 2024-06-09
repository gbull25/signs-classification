import logging
from io import BytesIO

import aioredis
import emoji
import numpy as np
import requests
from aiogram import Bot, F, Router, types
from aiogram.types import InputMediaPhoto, InputMediaVideo, Message, BufferedInputFile, FSInputFile
from aiogram.methods.send_video import SendVideo
from aiogram.utils.media_group import MediaGroupBuilder
from requests.exceptions import ConnectionError

router = Router()
logging.basicConfig(level=logging.INFO, format='%(asctime)s: [%(levelname)s] %(message)s')


# Хэндлер на альбом фотографий
@router.message(F.media_group_id, F.content_type.in_({'photo'}))
async def handle_albums(message: Message, album: list[Message], bot: Bot):
    #media_group = []
    #pred_result = []

    redis = await aioredis.from_url("redis://redis:5370")
    user_id = message.from_user.id
    model = await redis.get("user_id")
    if model == None:
        model = "cnn"
        await redis.set(user_id, "cnn")

    media_group = MediaGroupBuilder(caption="Результат детекции YOLO")

    for i, msg in enumerate(album):
        if msg.photo:

            file_id = msg.photo[-1].file_id
            #media_group.append(InputMediaPhoto(media=file_id))

            io = BytesIO()
            await bot.download(msg.photo[-1], destination=io)
            img = io.getvalue()

            try:

                response = requests.post(
                    "http://sign_classifier:80/detect_sign_photo",
                    files={'file_data': img}, params={"user_id": str(user_id), "suffix": ".jpg"}
                ).json()
                logging.info(f"Received a response with prediciton: {response}")
                #pred_result.append(InputMediaPhoto(media='attach://' + response['path']))
                media_group.add_photo(FSInputFile(path=response[0]["annotated_file_path"], filename="YOLO_result.jpg"))

            except ConnectionError as ce:

                logging.error(f"Connection refused error: {ce}")
                await message.reply("Кажется, в настоящее время сервис прилег :\( Попробуйте еще разок позже\!")
                return

    # Возвращаем альбом для удобства чтения результатов классификации
    #await message.answer_media_group(pred_result)
    await bot.send_media_group(chat_id=message.chat.id, media=media_group.build())
    # Возвращаем предсказания
    #i = 0
    #for pred in pred_result:
    #    i += 1
    #    await message.reply(f'На фотографии номер {i}:\n'
    #                        f'*{model.upper()}* считает, что знак {pred[0]} класса \(_{pred[1]}_\)\.')


# Хэндлер на одну фотографию
@router.message(F.photo)
async def predict_image(message: Message, bot: Bot):
    io = BytesIO()
    io = await bot.download(message.photo[-1], destination=io)
    img = io.getvalue()

    redis = await aioredis.from_url("redis://redis:5370")
    user_id = message.from_user.id
    model = await redis.get("user_id")
    if model == None:
        model = "cnn"
        await redis.set(user_id, "cnn")   

    try:

        response = requests.post(
                    "http://sign_classifier:80/detect_and_classify_signs",
                    files={'file_data': img}, params={"user_id": str(user_id), "suffix": ".jpg"}
        ).json()
        logging.info(f"Received a response with prediciton: {response}")

    except ConnectionError as ce:

        logging.error(f"Connection refused error: {ce}")
        await message.reply("Кажется, в настоящее время сервис прилег :\( Попробуйте еще разок позже\!")
        return

    ann_vid = FSInputFile(path=response[0]["annotated_file_path"], filename="YOLO_result.jpg")
    await message.reply_document(document=ann_vid, caption="Результат детекции YOLO")


# Хэндлер на видео
@router.message(F.video)
async def predict_video(message: Message, bot: Bot):
    io = BytesIO()
    io = await bot.download(message.video, destination=io)
    vid = io.getvalue()

    redis = await aioredis.from_url("redis://redis:5370")
    user_id = message.from_user.id
    model = await redis.get("user_id")
    if model == None:
        model = "cnn"
        await redis.set(user_id, "cnn")   
  
    await message.reply("Получил ваше видео, обрабатываю\.\.\.")

    try:

        response = requests.post(
                    "http://sign_classifier:80/detect_and_classify_signs",
                    files={'file_data': vid}, params={"user_id": str(user_id), "suffix": ".avi"}
        ).json()
        logging.info(f"Received a response with prediciton: {response}")

    except ConnectionError as ce:

        logging.error(f"Connection refused error: {ce}")
        await message.reply("Кажется, в настоящее время сервис прилег :\( Попробуйте еще разок позже\!")
        return

    ann_vid = FSInputFile(path=response[0]["annotated_file_path"], filename="YOLO_result.avi")
    await message.reply_document(document=ann_vid, caption="Результат детекции YOLO")


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
