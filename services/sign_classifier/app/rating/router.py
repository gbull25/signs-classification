from datetime import datetime

from app.auth.database import get_async_session
from app.auth.models import rating
from fastapi import APIRouter, Depends
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from .schemas import RatingAdd

router = APIRouter(
    prefix="/rating",
    tags=["Rating"]
)


@router.get("/current_rating")
async def get_rating(session: AsyncSession = Depends(get_async_session)):
    query = select(func.avg(rating.c.rating))
    result = await session.execute(query)
    return {
        "status": "success",
        "data": result.scalars().all()
    }


@router.post("/add_rating")
async def add_rating(new_operation: RatingAdd, session: AsyncSession = Depends(get_async_session)):
    insert_stmt = insert(rating).values(**new_operation.dict())
    do_update_stmt = insert_stmt.on_conflict_do_update(
        index_elements=['user_id'],
        set_=dict(rating=new_operation.rating, timestamp=datetime.utcnow())
    )

    await session.execute(do_update_stmt)
    await session.commit()
    return {"status": "success"}


@router.post("/delete_rating")
async def delete_rating(user_id: int, session: AsyncSession = Depends(get_async_session)):
    stmt = delete(rating).where(rating.c.user_id == user_id)
    await session.execute(stmt)
    await session.commit()
    return {"status": "success"}


@router.post("/get_rating_by_id")
async def get_rating_by_id(user_id: int, session: AsyncSession = Depends(get_async_session)):
    query = select(rating).where(rating.c.user_id == user_id)
    result = await session.execute(query)
    return {
        "status": "success",
        "data": result.all()[0]
    }
