import logging
from io import BytesIO

# import aiohttp
import requests
from aiogram import Bot, F, Router
from aiogram.types import InputMediaPhoto, Message
from requests.exceptions import ConnectionError

#from services.model import preprocessing

router = Router()
logging.basicConfig(level=logging.INFO, format='%(asctime)s: [%(levelname)s] %(message)s')


# Хэндлер на альбом фотографий
@router.message(F.media_group_id, F.content_type.in_({'photo'}))
async def handle_albums(message: Message, album: list[Message], bot: Bot):
    media_group = []
    hog_pred = []
    sift_pred = []
    cnn_pred = []

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

                # HOG
                data_hog = requests.post("http://sign_classifier:80/predict/sign_hog", files={'file': im}).json()
                hog_pred.append((data_hog['sign_class'], data_hog['sign_description']))

                # SIFT
                data_sift = requests.post("http://sign_classifier:80/predict/sign_sift", files={'file': im}).json()
                sift_pred.append((data_sift['sign_class'], data_sift['sign_description']))

                # CNN
                data_cnn = requests.post("http://sign_classifier:80/predict/sign_cnn", files={'file': im}).json()
                cnn_pred.append((data_cnn['sign_class'], data_cnn['sign_description']))

            except ConnectionError as ce:

                logging.error(f"Connection refused error: {ce}")
                await message.reply("Кажется, в настоящее время сервис прилег :\( Попробуйте еще разок позже\!")
                return

    # aiohttp WIP
    # async with aiohttp.ClientSession() as session:
    #     res = await session.post("http://sign_classifier:80/predict/signs_cnn", data=request_data)

    # await message.answer(res.status)

    data_class = {'hog_class': hog_pred, 'sift_class': sift_pred, 'cnn_class': cnn_pred}

    # Возвращаем альбом для удобства чтения результатов классификации
    await message.answer_media_group(media_group)
    # Возвращаем предсказания
    i = 0
    for hog, sift, cnn in zip(data_class['hog_class'], data_class['sift_class'], data_class['cnn_class']):
        i += 1
        await message.reply(f'На фотографии номер {i}:\n'
                            f'*HOG SVM* считает, что знак {hog[0]} класса \(_{hog[1]}_\),\n'
                            f'*SIFT SVM* считает, что знак {sift[0]} класса \(_{sift[1]}_\),\n'
                            f'*CNN* считает, что знак {cnn[0]} класса \(_{cnn[1]}_\)\.')


# Хэндлер на одну фотографию
@router.message(F.photo)
async def predict_image(message: Message, bot: Bot):
    io = BytesIO()
    io = await bot.download(message.photo[-1], destination=io)
    im = io.getvalue()

    try:

        data_hog = requests.post("http://sign_classifier:80/predict/sign_hog", files={'file': im}).json()
        data_sift = requests.post("http://sign_classifier:80/predict/sign_sift", files={'file': im}).json()
        data_cnn = requests.post("http://sign_classifier:80/predict/sign_cnn", files={'file': im}).json()

    except ConnectionError as ce:

        logging.error(f"Connection refused error: {ce}")
        await message.reply("Кажется, в настоящее время сервис прилег :\( Попробуйте еще разок позже\!")
        return

    await message.reply(f'*HOG SVM* считает, что этот знак '
            f'{data_hog["sign_class"]} класса \(_{data_hog["sign_description"]}_\),\n'
            f'*SIFT SVM* считает, что этот знак '
            f'{data_sift["sign_class"]} класса \(_{data_sift["sign_description"]}_\),\n'
            f'*CNN* считает, что этот знак '
            f'{data_cnn["sign_class"]} класса \(_{data_cnn["sign_description"]}_\)\.')