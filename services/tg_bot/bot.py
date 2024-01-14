import asyncio
import logging

from aiogram import Bot, Dispatcher
from config_reader import config
from handlers import menu, predictions
from middleware import Album_Middleware


async def main():
    # Включаем логирование, чтобы не пропустить важные сообщения
    logging.basicConfig(level=logging.INFO)
    # Объект бота
    bot = Bot(token=config.bot_token.get_secret_value(), parse_mode='MarkdownV2')
    # Объект диспетчера
    dp = Dispatcher()
    # Добавляем Middleware на альбом изображений
    dp.message.middleware(Album_Middleware())
    # Добавляем роутеры
    dp.include_routers(menu.router, predictions.router)    

    # Запуск процесса поллинга новых апдейтов
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())