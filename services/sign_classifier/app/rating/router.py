from app.auth.database import get_async_session
from app.auth.models import rating
from fastapi import APIRouter, Depends
from sqlalchemy import func, insert, select
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
    stmt = insert(rating).values(**new_operation.dict())
    await session.execute(stmt)
    await session.commit()
    return {"status": "success"}
