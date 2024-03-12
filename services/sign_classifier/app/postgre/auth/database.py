from typing import AsyncGenerator
from datetime import datetime

from fastapi import Depends
from fastapi_users.db import SQLAlchemyBaseUserTable, SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Boolean, Integer, TIMESTAMP, ForeignKey
from config import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER
from models.models import role

DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


class Base(DeclarativeBase):
    pass


class User(SQLAlchemyBaseUserTable[int], Base):
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # email: Mapped[str] = mapped_column(String, nullable=False) дублируется
    username: Mapped[str] = mapped_column(String, nullable=False)
    # password: Mapped[str] = mapped_column(String, nullable=False) убираем в пользу hashed_password
    registered_at: Mapped[datetime.timestamp] = mapped_column(TIMESTAMP, default=datetime.utcnow)
    role_id: Mapped[int] = mapped_column(Integer, ForeignKey(role.c.id))

    email: Mapped[str] = mapped_column(
        String(length=320), unique=True, index=True, nullable=False
    )
    hashed_password: Mapped[str] = mapped_column(
        String(length=1024), nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )


engine = create_async_engine(DATABASE_URL)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)

# Убираем поскольку используем alembic
# async def create_db_and_tables():
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)
#

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session


# Depends отвечает за dependency injection
async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, User)
