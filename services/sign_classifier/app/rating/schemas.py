from pydantic import BaseModel


class RatingGet(BaseModel):
    rating: int

    class Config:
        orm_mode = True


class RatingAdd(BaseModel):
    user_id: int
    rating: int
