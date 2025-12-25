import os
import cv2
import numpy as np

from infer_shelf import predict_shelf_from_array

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart

import asyncio
from aiogram.fsm.storage.memory import MemoryStorage

from config import PAVILIONS
from dotenv import load_dotenv

load_dotenv()

CONFIG_KEY = "shelf_full_1.jpg"

TELEGRAM_TOKEN = os.getenv('BOT_TOKEN')

if not TELEGRAM_TOKEN:
    raise ValueError("BOT_TOKEN is not set")

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(storage=MemoryStorage())


@dp.message(CommandStart())
async def start_command(message: Message):
    text = (
        "Привет! Я помогу проверить витрину.\n\n"
        "1️⃣ Пришли фото витрины.\n"
        "2️⃣ В подписи к фото укажи ID павильона:\n"
        "   1 — Карусель\n"
        "   2 — Пятёрочка\n"
        "   3 — Таллин\n"
        "   4 — Якорь\n"
        "Я скажу, ОК ли витрина."
    )
    await message.answer(text, parse_mode=ParseMode.HTML)

@dp.message()
async def handle_photo(message: Message):
    print(f"New message from {message.from_user.id}, has_photo={bool(message.photo)}, caption={message.caption!r}")

    if not message.photo:
        await message.reply(
            "Пришли, пожалуйста, фото витрины с подписью — номер павильона (1–4).",
            parse_mode=ParseMode.HTML,
        )
        return

    caption = (message.caption or '').strip()
    if not caption:
        await message.reply(
            "В подписи к фото укажи ID павильона (1–4), например: <b>2</b>.",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        pavilion_id = int(caption)
    except ValueError:
        await message.reply(
            "ID павильона должен быть числом от 1 до 4.\nНапример: <b>1</b> или <b>3</b>.",
            parse_mode=ParseMode.HTML,
        )
        return
    
    if pavilion_id not in PAVILIONS:
        await message.reply(
            f"Неизвестный ID павильона: {pavilion_id}. Доступные: {', '.join(str(k) for k in PAVILIONS.keys())}.",
            parse_mode=ParseMode.HTML,
        )
        return

    pavilion_name = PAVILIONS[pavilion_id]
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    file_bytes = await bot.download_file(file.file_path)
    
    file_bytes.seek(0)
    data = np.frombuffer(file_bytes.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if img is None:
        await message.reply(
            "Ошибка при загрузке фото. Попробуйте ещё раз.",
            parse_mode=ParseMode.HTML,
        )
        return
    
    try:
        result = predict_shelf_from_array(img, CONFIG_KEY)
    except Exception as e:
        print(f"Error in predict_shelf_from_array: {e}")
        await message.reply("Ошибка при анализе витрины, попробуй ещё раз позже.")
        return

    status = result.get('status', 'unknown')
    zones = {k: v for k, v in result.items() if k != "status"}
    zones_text = ", ".join(f"{name}: {label}" for name, label in zones.items()) or "нет данных по зонам"

    if status == 'ok':
        text = (
            f"✅ Витрина ОК для павильона <b>{pavilion_name}</b> (ID={pavilion_id}).\n"
            f"Зоны: {zones_text}"
        )
    elif status == "not ok":
        text = (
            f"❌ Витрина НЕ ОК для павильона <b>{pavilion_name}</b> (ID={pavilion_id}).\n"
            f"Проблемные зоны: {zones_text}"
        )
    else:
        text = (
            f"⚠️ Не удалось однозначно определить статус витрины "
            f"для павильона <b>{pavilion_name}</b> (ID={pavilion_id}).\n"
            f"Сырые данные: {zones_text}"
        )

    await message.reply(text, parse_mode=ParseMode.HTML)



    await message.reply(
        f"Окей, павильон: <b>{pavilion_name}</b> (ID={pavilion_id}). "
        f"На следующем шаге проверю витрину.",
        parse_mode=ParseMode.HTML,
    )

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
