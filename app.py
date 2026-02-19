import asyncio
import hashlib
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any

import aiosqlite
import qrcode
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import aiohttp
import uvicorn


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("raffle_bot")


@dataclass
class Settings:
    bot_token: str = os.getenv("BOT_TOKEN", "")
    admin_ids: set[int] = None
    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "")
    terminal_key: str = os.getenv("TBANK_TERMINAL_KEY", "")
    password: str = os.getenv("TBANK_PASSWORD", "")
    db_path: str = os.getenv("DB_PATH", "/app/data/bot.db")
    ticket_price: int = int(os.getenv("TICKET_PRICE", "999"))
    ticket_max: int = int(os.getenv("TICKET_MAX", "2599"))
    big_purchase_tickets_threshold: int | None = (
        int(v) if (v := os.getenv("BIG_PURCHASE_TICKETS_THRESHOLD")) else None
    )
    big_purchase_amount_threshold_rub: int | None = (
        int(v) if (v := os.getenv("BIG_PURCHASE_AMOUNT_THRESHOLD_RUB")) else None
    )
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8080"))

    def __post_init__(self):
        admin_raw = os.getenv("ADMIN_IDS", "")
        self.admin_ids = {int(x.strip()) for x in admin_raw.split(",") if x.strip()}


settings = Settings()
if not settings.bot_token:
    raise RuntimeError("BOT_TOKEN is not set")


class BuyStates(StatesGroup):
    waiting_qty = State()


class ManualStates(StatesGroup):
    waiting_phone = State()
    waiting_qty = State()


class Service:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.bot: Bot | None = None

    async def init_db(self):
        os.makedirs(os.path.dirname(self.settings.db_path), exist_ok=True)
        async with aiosqlite.connect(self.settings.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    tg_user_id INTEGER PRIMARY KEY,
                    tg_username TEXT,
                    phone TEXT,
                    created_ts REAL,
                    updated_ts REAL
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS payments (
                    payment_id TEXT PRIMARY KEY,
                    tg_user_id INTEGER,
                    amount_rub INTEGER,
                    ticket_qty INTEGER,
                    order_id TEXT,
                    status TEXT,
                    created_ts REAL,
                    confirmed_ts REAL
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS tickets (
                    number INTEGER PRIMARY KEY,
                    payment_id TEXT,
                    tg_user_id INTEGER,
                    allocated_ts REAL,
                    source_admin_id INTEGER
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            await db.execute(
                "INSERT OR IGNORE INTO bot_state(key, value) VALUES('registration_open', '1')"
            )
            await db.commit()

    async def db_fetchone(self, query: str, params: tuple = ()):
        async with aiosqlite.connect(self.settings.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(query, params)
            row = await cur.fetchone()
            return row

    async def db_fetchall(self, query: str, params: tuple = ()):
        async with aiosqlite.connect(self.settings.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(query, params)
            return await cur.fetchall()

    async def db_execute(self, query: str, params: tuple = ()):
        async with aiosqlite.connect(self.settings.db_path) as db:
            await db.execute(query, params)
            await db.commit()

    async def remaining_tickets(self) -> int:
        row = await self.db_fetchone("SELECT COUNT(*) AS c FROM tickets")
        return self.settings.ticket_max - int(row["c"])

    async def registration_open(self) -> bool:
        row = await self.db_fetchone("SELECT value FROM bot_state WHERE key='registration_open'")
        return row and row["value"] == "1"

    def tbank_token(self, data: dict[str, Any]) -> str:
        payload = {k: v for k, v in data.items() if k != "Token" and not isinstance(v, (dict, list))}
        payload["Password"] = self.settings.password
        digest = "".join(str(payload[k]) for k in sorted(payload))
        return hashlib.sha256(digest.encode()).hexdigest()

    async def tbank_request(self, method: str, body: dict[str, Any]) -> dict[str, Any]:
        if not self.settings.terminal_key or not self.settings.password:
            raise RuntimeError("TBANK_TERMINAL_KEY/TBANK_PASSWORD are not set")
        body["TerminalKey"] = self.settings.terminal_key
        body["Token"] = self.tbank_token(body)
        async with aiohttp.ClientSession() as session:
            async with session.post(f"https://securepay.tinkoff.ru/v2/{method}", json=body, timeout=30) as resp:
                data = await resp.json()
                if not data.get("Success"):
                    raise RuntimeError(f"T-Bank {method} failed: {data}")
                return data

    async def create_payment(self, tg_user_id: int, qty: int) -> dict[str, Any]:
        amount_rub = qty * self.settings.ticket_price
        order_id = str(uuid.uuid4())
        init = await self.tbank_request(
            "Init",
            {
                "Amount": amount_rub * 100,
                "OrderId": order_id,
                "NotificationURL": f"{self.settings.public_base_url}/tbank/webhook",
                "Description": f"Raffle tickets x{qty}",
            },
        )
        payment_id = str(init["PaymentId"])
        qr = await self.tbank_request("GetQr", {"PaymentId": payment_id, "DataType": "PAYLOAD"})
        await self.db_execute(
            """
            INSERT OR REPLACE INTO payments(payment_id, tg_user_id, amount_rub, ticket_qty, order_id, status, created_ts, confirmed_ts)
            VALUES(?, ?, ?, ?, ?, 'NEW', ?, NULL)
            """,
            (payment_id, tg_user_id, amount_rub, qty, order_id, time.time()),
        )
        logger.info("created payment_id=%s user_id=%s qty=%s amount=%s", payment_id, tg_user_id, qty, amount_rub)
        return {"payment_id": payment_id, "payload": qr["Data"], "payment_url": init.get("PaymentURL", "")}

    async def check_state(self, payment_id: str) -> str:
        resp = await self.tbank_request("GetState", {"PaymentId": payment_id})
        return resp.get("Status", "UNKNOWN")

    async def finalize_payment(self, payment_id: str) -> tuple[bool, list[int], int]:
        async with aiosqlite.connect(self.settings.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            p = await (await db.execute("SELECT * FROM payments WHERE payment_id=?", (payment_id,))).fetchone()
            if not p:
                await db.rollback()
                return False, [], await self.remaining_tickets()
            already = await (await db.execute("SELECT COUNT(*) as c FROM tickets WHERE payment_id=?", (payment_id,))).fetchone()
            if already["c"] > 0:
                await db.commit()
                nums = await self.db_fetchall("SELECT number FROM tickets WHERE payment_id=? ORDER BY number", (payment_id,))
                rem = await self.remaining_tickets()
                return False, [r["number"] for r in nums], rem
            rem_row = await (await db.execute("SELECT COUNT(*) as c FROM tickets")).fetchone()
            remaining = self.settings.ticket_max - rem_row["c"]
            if p["ticket_qty"] > remaining:
                await db.rollback()
                raise RuntimeError("Not enough tickets remaining")
            free = await (await db.execute(
                """
                SELECT n AS number
                FROM (WITH RECURSIVE seq(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM seq WHERE n < ?) SELECT n FROM seq)
                WHERE n NOT IN (SELECT number FROM tickets)
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (self.settings.ticket_max, p["ticket_qty"]),
            )).fetchall()
            ts = time.time()
            numbers = [r["number"] for r in free]
            if len(numbers) != p["ticket_qty"]:
                await db.rollback()
                raise RuntimeError("Could not allocate required amount of tickets")
            for n in numbers:
                await db.execute(
                    "INSERT INTO tickets(number, payment_id, tg_user_id, allocated_ts, source_admin_id) VALUES(?, ?, ?, ?, NULL)",
                    (n, payment_id, p["tg_user_id"], ts),
                )
            await db.execute("UPDATE payments SET status='CONFIRMED', confirmed_ts=? WHERE payment_id=?", (ts, payment_id))
            await db.commit()

        remaining_now = await self.remaining_tickets()
        logger.info("finalized payment_id=%s tickets=%s remaining=%s", payment_id, numbers, remaining_now)
        return True, sorted(numbers), remaining_now

    async def notify_big_purchase(self, payment_id: str, ticket_qty: int, amount_rub: int, remaining: int):
        trigger = False
        if self.settings.big_purchase_tickets_threshold is not None and ticket_qty >= self.settings.big_purchase_tickets_threshold:
            trigger = True
        if self.settings.big_purchase_amount_threshold_rub is not None and amount_rub >= self.settings.big_purchase_amount_threshold_rub:
            trigger = True
        if not trigger:
            return
        row = await self.db_fetchone(
            "SELECT p.tg_user_id, u.tg_username, u.phone FROM payments p LEFT JOIN users u ON u.tg_user_id=p.tg_user_id WHERE p.payment_id=?",
            (payment_id,),
        )
        text = (
            f"Крупная покупка\n"
            f"Дата/время: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"tg_user_id: {row['tg_user_id']}\n"
            f"username: @{row['tg_username'] or '-'}\n"
            f"телефон: {row['phone'] or '-'}\n"
            f"ticket_qty: {ticket_qty}\n"
            f"amount_rub: {amount_rub}\n"
            f"PaymentId: {payment_id}\n"
            f"остаток: {remaining}"
        )
        for aid in self.settings.admin_ids:
            try:
                await self.bot.send_message(aid, text)
            except Exception:
                logger.exception("failed notify admin %s", aid)
        logger.info("big purchase notified payment_id=%s", payment_id)


svc = Service(settings)
router = Router()


def parse_positive_int(raw: str) -> int | None:
    try:
        value = int(raw.strip())
        if value <= 0:
            return None
        return value
    except Exception:
        return None


def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="💳 Купить номерки", callback_data="buy")]])


@router.message(Command("start"))
async def start(message: Message):
    if not await svc.registration_open() and message.from_user.id not in settings.admin_ids:
        await message.answer("Регистрация временно закрыта")
        return
    user = await svc.db_fetchone("SELECT phone FROM users WHERE tg_user_id=?", (message.from_user.id,))
    if not user or not user["phone"]:
        kb = ReplyKeyboardMarkup(
            keyboard=[[KeyboardButton(text="📱 Отправить номер", request_contact=True)]],
            resize_keyboard=True,
            one_time_keyboard=True,
        )
        await message.answer("Отправьте ваш номер телефона", reply_markup=kb)
        return
    remaining = await svc.remaining_tickets()
    await message.answer(
        f"Стоимость: {settings.ticket_price} руб.\nВсего номерков: {settings.ticket_max}\nОсталось: {remaining}",
        reply_markup=main_menu(),
    )


@router.message(F.contact)
async def save_contact(message: Message):
    ts = time.time()
    await svc.db_execute(
        """
        INSERT INTO users(tg_user_id, tg_username, phone, created_ts, updated_ts)
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(tg_user_id) DO UPDATE SET tg_username=excluded.tg_username, phone=excluded.phone, updated_ts=excluded.updated_ts
        """,
        (message.from_user.id, message.from_user.username, message.contact.phone_number, ts, ts),
    )
    logger.info("saved phone user_id=%s", message.from_user.id)
    remaining = await svc.remaining_tickets()
    await message.answer(
        f"Телефон сохранен.\nСтоимость: {settings.ticket_price} руб.\nВсего: {settings.ticket_max}\nОсталось: {remaining}",
        reply_markup=main_menu(),
    )


@router.callback_query(F.data == "buy")
async def buy_prompt(callback: CallbackQuery, state: FSMContext):
    await callback.message.answer("Введите количество номерков:")
    await state.set_state(BuyStates.waiting_qty)
    await callback.answer()


@router.message(BuyStates.waiting_qty)
async def buy_qty(message: Message, state: FSMContext):
    qty = parse_positive_int(message.text or "")
    if qty is None:
        await message.answer("Введите целое число")
        return
    remaining = await svc.remaining_tickets()
    if qty > remaining:
        await message.answer(f"Некорректное количество. Доступно: {remaining}")
        return
    try:
        payment = await svc.create_payment(message.from_user.id, qty)
    except Exception:
        logger.exception("failed create payment user_id=%s qty=%s", message.from_user.id, qty)
        await message.answer("Не удалось создать платеж, попробуйте позже")
        return
    img = qrcode.make(payment["payload"])
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    rows = [[InlineKeyboardButton(text="✅ Оплатил(а), проверить", callback_data=f"check:{payment['payment_id']}")]]
    if payment.get("payment_url"):
        rows.append([InlineKeyboardButton(text="💳 Оплатить картой", url=payment["payment_url"])])
    kb = InlineKeyboardMarkup(inline_keyboard=rows)
    await message.answer_photo(
        BufferedInputFile(buf.getvalue(), filename="sbp_qr.png"),
        caption=(
            "Предпочтительный способ: СБП (по QR).\n"
            "Если удобнее, можно оплатить картой по кнопке ниже.\n"
            f"Количество: {qty}\nСумма: {qty * settings.ticket_price} руб."
        ),
        reply_markup=kb,
    )
    await state.clear()


@router.callback_query(F.data.startswith("check:"))
async def check_payment_callback(callback: CallbackQuery):
    payment_id = callback.data.split(":", 1)[1]
    p = await svc.db_fetchone("SELECT * FROM payments WHERE payment_id=?", (payment_id,))
    if not p or p["tg_user_id"] != callback.from_user.id:
        await callback.answer("Платеж не найден", show_alert=True)
        return
    try:
        status = await svc.check_state(payment_id)
    except Exception:
        logger.exception("fallback check failed payment_id=%s", payment_id)
        await callback.message.answer("Не удалось проверить оплату. Повторите позже")
        await callback.answer()
        return
    logger.info("fallback check payment_id=%s status=%s", payment_id, status)
    if status == "CONFIRMED":
        created, nums, rem = await svc.finalize_payment(payment_id)
        await callback.message.answer(f"Оплата подтверждена. Ваши номерки: {', '.join(map(str, nums))}\nОсталось: {rem}")
        await svc.notify_big_purchase(payment_id, p["ticket_qty"], p["amount_rub"], rem)
    else:
        await callback.message.answer(f"Текущий статус: {status}")
    await callback.answer()


@router.message(Command("check_payment"))
async def check_payment_cmd(message: Message):
    p = await svc.db_fetchone("SELECT payment_id FROM payments WHERE tg_user_id=? ORDER BY created_ts DESC LIMIT 1", (message.from_user.id,))
    if not p:
        await message.answer("Платежи не найдены")
        return
    try:
        status = await svc.check_state(p["payment_id"])
    except Exception:
        logger.exception("check_payment command failed payment_id=%s", p["payment_id"])
        await message.answer("Не удалось проверить оплату. Повторите позже")
        return
    if status == "CONFIRMED":
        created, nums, rem = await svc.finalize_payment(p["payment_id"])
        await message.answer(f"Оплата подтверждена. Ваши номерки: {', '.join(map(str, nums))}\nОсталось: {rem}")
        full_payment = await svc.db_fetchone("SELECT ticket_qty, amount_rub FROM payments WHERE payment_id=?", (p["payment_id"],))
        if full_payment:
            await svc.notify_big_purchase(p["payment_id"], full_payment["ticket_qty"], full_payment["amount_rub"], rem)
    else:
        await message.answer(f"Статус платежа {p['payment_id']}: {status}")


def is_admin(uid: int) -> bool:
    return uid in settings.admin_ids


@router.message(Command("stats"))
async def stats(message: Message):
    if not is_admin(message.from_user.id):
        return
    reg = await svc.registration_open()
    users = await svc.db_fetchone("SELECT COUNT(*) c FROM users")
    issued = await svc.db_fetchone("SELECT COUNT(*) c FROM tickets")
    pending = await svc.db_fetchone("SELECT COUNT(*) c FROM payments WHERE status != 'CONFIRMED'")
    rem = settings.ticket_max - issued["c"]
    await message.answer(
        f"Регистрация: {'открыта' if reg else 'закрыта'}\n"
        f"Пользователей: {users['c']}\nВыдано: {issued['c']}\nОстаток: {rem}\nОжидают оплат: {pending['c']}"
    )


@router.message(Command("stop_registration"))
async def stop_registration(message: Message):
    if not is_admin(message.from_user.id):
        return
    await svc.db_execute("UPDATE bot_state SET value='0' WHERE key='registration_open'")
    await message.answer("Регистрация закрыта")


@router.message(Command("resume_registration"))
async def resume_registration(message: Message):
    if not is_admin(message.from_user.id):
        return
    await svc.db_execute("UPDATE bot_state SET value='1' WHERE key='registration_open'")
    await message.answer("Регистрация открыта")


@router.message(Command("manual_allocate"))
async def manual_allocate(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    await state.set_state(ManualStates.waiting_phone)
    await message.answer("Введите телефон участника:")


@router.message(ManualStates.waiting_phone)
async def manual_phone(message: Message, state: FSMContext):
    await state.update_data(phone=message.text.strip())
    await state.set_state(ManualStates.waiting_qty)
    await message.answer("Введите количество билетов:")


@router.message(ManualStates.waiting_qty)
async def manual_qty(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    data = await state.get_data()
    phone = data["phone"]
    qty = parse_positive_int(message.text or "")
    if qty is None:
        await message.answer("Введите корректное целое число > 0")
        return
    payment_id = f"manual-{message.from_user.id}-{int(time.time())}-{random.randint(100,999)}"
    amount = qty * settings.ticket_price
    ts = time.time()
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("BEGIN IMMEDIATE")
        issued_row = await (await db.execute("SELECT COUNT(*) as c FROM tickets")).fetchone()
        remaining = settings.ticket_max - issued_row["c"]
        if qty > remaining:
            await db.rollback()
            await message.answer(f"Недостаточно свободных номерков. Осталось: {remaining}")
            return
        await db.execute(
            "INSERT INTO payments(payment_id, tg_user_id, amount_rub, ticket_qty, order_id, status, created_ts, confirmed_ts) VALUES(?, NULL, ?, ?, ?, 'CONFIRMED', ?, ?)",
            (payment_id, amount, qty, payment_id, ts, ts),
        )
        free = await (await db.execute(
            """
            SELECT n AS number
            FROM (WITH RECURSIVE seq(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM seq WHERE n < ?) SELECT n FROM seq)
            WHERE n NOT IN (SELECT number FROM tickets)
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (settings.ticket_max, qty),
        )).fetchall()
        nums = [r["number"] for r in free]
        if len(nums) != qty:
            await db.rollback()
            await message.answer("Не удалось выделить запрошенное количество номерков")
            return
        for n in nums:
            await db.execute(
                "INSERT INTO tickets(number, payment_id, tg_user_id, allocated_ts, source_admin_id) VALUES(?, ?, NULL, ?, ?)",
                (n, payment_id, ts, message.from_user.id),
            )
        await db.commit()
    await message.answer(f"Выданы номерки для {phone}: {', '.join(map(str, sorted(nums)))}")
    await state.clear()


app = FastAPI()


@app.post("/tbank/webhook")
async def tbank_webhook(request: Request):
    payload = await request.json()
    payment_id = str(payload.get("PaymentId", ""))
    status = payload.get("Status")
    token = payload.get("Token")
    if token != svc.tbank_token(payload):
        return JSONResponse({"ok": False, "error": "invalid token"}, status_code=400)
    logger.info("webhook payment_id=%s status=%s", payment_id, status)
    if status == "CONFIRMED":
        p = await svc.db_fetchone("SELECT * FROM payments WHERE payment_id=?", (payment_id,))
        if p:
            created, nums, rem = await svc.finalize_payment(payment_id)
            if svc.bot and p["tg_user_id"]:
                await svc.bot.send_message(p["tg_user_id"], f"Оплата подтверждена. Ваши номерки: {', '.join(map(str, nums))}\nОсталось: {rem}")
            await svc.notify_big_purchase(payment_id, p["ticket_qty"], p["amount_rub"], rem)
    return {"ok": True}


async def main():
    logger.info("starting service")
    await svc.init_db()
    bot = Bot(settings.bot_token)
    svc.bot = bot
    dp = Dispatcher()
    dp.include_router(router)

    config = uvicorn.Config(app, host=settings.app_host, port=settings.app_port, log_level="info")
    server = uvicorn.Server(config)
    webhook_task = asyncio.create_task(server.serve())
    try:
        await dp.start_polling(bot)
    finally:
        webhook_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
