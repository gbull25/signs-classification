from io import BytesIO

from aiogram import Bot, F, Router
from aiogram.types import InputMediaPhoto, Message

from services.model import preprocessing
from services.tg_bot.config_reader import config

bot = Bot(token=config.bot_token.get_secret_value(), parse_mode='MarkdownV2')
router = Router()


# Хэндлер на альбом фотографий
@router.message(F.media_group_id, F.content_type.in_({'photo'}))
async def handle_albums(message: Message, album: list[Message]):
    media_group = []
    hog_pred = []
    sift_pred = []
    for msg in album:
        if msg.photo:
            file_id = msg.photo[-1].file_id
            media_group.append(InputMediaPhoto(media=file_id))

            io = BytesIO()
            await bot.download(msg.photo[-1], destination=io)
            im = io.getvalue()
            data_hog = preprocessing.predict_hog_image(im)
            hog_pred.append((data_hog['sign_class'], data_hog['sign_description']))
            data_sift = preprocessing.predict_sift_image(im)
            sift_pred.append((data_sift['sign_class'], data_sift['sign_description']))

    data_class = {'hog_class': hog_pred, 'sift_class': sift_pred}

    # Возвращаем альбом для удобства чтения результатов классификации
    await message.answer_media_group(media_group) 
    # Возвращаем предсказания
    i = 0
    for hog, sift in zip(data_class['hog_class'], data_class['sift_class']):
        i += 1
        await message.reply(f'На фотографии номер {i}:\n'
                           f'*HOG SVM* считает, что знак {hog[0]} класса \(_{hog[1]}_\),\n'
                           f'*SIFT SVM* считает, что знак {sift[0]} класса \(_{sift[1]}_\)\.') 


# Хэндлер на одну фотографию
@router.message(F.photo)
async def predict_image(message: Message):
    io = BytesIO()
    await bot.download(message.photo[-1], destination=io)
    im = io.getvalue()
    data_hog = preprocessing.predict_hog_image(im)
    data_sift = preprocessing.predict_sift_image(im)
    await message.reply(f'*HOG SVM* считает, что этот знак '
                        f'{data_hog["sign_class"]} класса \(_{data_hog["sign_description"]}_\),\n'
                        f'*SIFT SVM* считает, что этот знак '
                        f'{data_sift["sign_class"]} класса \(_{data_sift["sign_description"]}_\)\.')
