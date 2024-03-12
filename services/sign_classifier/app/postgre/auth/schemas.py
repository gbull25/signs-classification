from fastapi_users import schemas
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar
from pydantic import BaseModel, ConfigDict, EmailStr


class UserRead(schemas.BaseUser[int]):
    id: int
    email: str
    username: str
    # пароль здесь выводить, разумеется, нельзя
    role_id: int
    is_active: bool = True
    is_superuser: bool = False


class UserCreate(schemas.BaseUserCreate):
    username: str
    email: str
    password: str
    role_id: int
    is_active: Optional[bool] = True
    is_superuser: Optional[bool] = False

# удалим для простоты
# class UserUpdate(schemas.BaseUserUpdate):
#     pass