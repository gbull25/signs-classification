from io import BytesIO
from aiogram import Router, F, Bot
from aiogram.types import Message, InputMediaPhoto
from services.tg_bot.config_reader import config
from services.model import preprocessing

bot = Bot(token=config.bot_token.get_secret_value(), parse_mode='MarkdownV2')
router = Router()

# Хэндлер на альбом фотографий
@router.message(F.media_group_id, F.content_type.in_({'photo'}))
async def handle_albums(message: Message, album: list[Message]):
    media_group = []
    class_pred = []
    for msg in album:
        if msg.photo:
            file_id = msg.photo[-1].file_id
            media_group.append(InputMediaPhoto(media=file_id))

            io = BytesIO()
            await bot.download(msg.photo[-1], destination=io)
            im = io.getvalue()
            data = preprocessing.predict_hog_image(im)
            class_pred.append(data['sign class'])

    await message.answer_media_group(media_group) 

    i = 0
    for sign in class_pred:
        i += 1
        await message.reply(f"На фотографии номер {i} знак точно {sign} класса") 


# Хэндлер на одну фотографию
@router.message(F.photo)
async def predict_image(message: Message):
    io = BytesIO()
    await bot.download(message.photo[-1], destination=io)
    im = io.getvalue()
    data = preprocessing.predict_hog_image(im)
    await message.reply(f"Этот знак точно {data['sign class']} класса")