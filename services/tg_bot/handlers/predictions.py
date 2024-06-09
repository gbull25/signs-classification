import logging
from io import BytesIO
from collections import defaultdict
from tempfile import NamedTemporaryFile

import aioredis
import emoji
import numpy as np
import requests
import csv
import pathlib
from aiogram import Bot, F, Router, types
from aiogram.types import InputMediaPhoto, InputMediaVideo, Message, BufferedInputFile, FSInputFile
from aiogram.methods.send_video import SendVideo
from aiogram.utils.media_group import MediaGroupBuilder
from requests.exceptions import ConnectionError

router = Router()
logging.basicConfig(level=logging.INFO, format='%(asctime)s: [%(levelname)s] %(message)s')


async def form_csv(classification_data: dict):
    keys = classification_data[0].keys()

    with NamedTemporaryFile(delete=False, suffix=".csv", mode="w") as tmp:
        dict_writer = csv.DictWriter(tmp, keys)
        dict_writer.writeheader()
        dict_writer.writerows(classification_data)
        csv_name = tmp.name

    return csv_name


# Хэндлер на альбом фотографий
@router.message(F.media_group_id, F.content_type.in_({'photo'}))
async def handle_albums(message: Message, album: list[Message], bot: Bot):
    redis = await aioredis.from_url("redis://redis:5370")
    user_id = message.from_user.id
    model = await redis.get("user_id")
    if model == None:
        model = "cnn"
        await redis.set(user_id, "cnn")

    media_group_photos = MediaGroupBuilder(caption="Результат детекции YOLO")
    media_group_csvs = MediaGroupBuilder()

    # Storing csv's paths to delete later
    csv_paths = []

    for i, msg in enumerate(album):
        if msg.photo:

            # file_id = msg.photo[-1].file_id

            io = BytesIO()
            await bot.download(msg.photo[-1], destination=io)
            img = io.getvalue()

            try:

                response = requests.post(
                    "http://sign_classifier:80/detect_and_classify_signs",
                    files={'file_data': img}, params={"user_id": str(user_id), "suffix": ".jpg"}
                ).json()
                logging.info(f"Received a response with prediciton: {response}")
                res_csv_path = await form_csv(response)
                media_group_photos.add_photo(FSInputFile(path=response[0]["result_file_path"], filename=f"YOLO_result_{i}.jpg"))
                media_group_csvs.add_document(FSInputFile(path=res_csv_path, filename=f"YOLO_result_text_{i}.csv"))

            except ConnectionError as ce:

                logging.error(f"Connection refused error: {ce}")
                await message.reply("Кажется, в настоящее время сервис прилег :\( Попробуйте еще разок позже\!")
                return

    # Возвращаем альбом для удобства чтения результатов классификации
    await bot.send_media_group(chat_id=message.chat.id, media=media_group_photos.build())
    await bot.send_media_group(chat_id=message.chat.id, media=media_group_csvs.build())

    for p in csv_paths:
        pathlib.Path(p).unlink()


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

    ann_vid = FSInputFile(path=response[0]["result_file_path"], filename="YOLO_result.jpg")
    res_csv_path = await form_csv(response)
    res_csv = FSInputFile(path=res_csv_path, filename="YOLO_result_text.csv")

    await message.reply_document(document=ann_vid, caption="Результат детекции YOLO")
    await message.reply_document(document=res_csv, caption="CSV с результатами")

    pathlib.Path(res_csv_path).unlink()


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

    res_csv_path = await form_csv(response)
    res_csv = FSInputFile(path=res_csv_path, filename="YOLO_result_text.csv")

    await message.reply_document(document=ann_vid, caption="Результат детекции YOLO")
    await message.reply_document(document=res_csv, caption="CSV с результатами")

    pathlib.Path(res_csv_path).unlink()


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
