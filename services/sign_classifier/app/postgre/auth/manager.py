from typing import Optional

from fastapi import Depends, Request
from fastapi_users import BaseUserManager, IntegerIDMixin

from auth.database import User, get_user_db

SECRET = "SECRET"  # здесь другой ключ для сброса пароля и верификации


# UUIDIDMixin -> IntegerIDMixin, uuid.UUID]-> int
class UserManager(IntegerIDMixin, BaseUserManager[User, int]):
    # загляните в BaseUserManager и посмотрите на функцию create
    # после этого станет понятно, почему, например, где-то у нас поле password, а потом оно становится hashed_password
    # можем скопировать функцию create сюда и видоизменить её, например, добавить user_dict["role_id"] = 1
    # чтобы по умолчанию выдавать всем новым пользователям роль 1
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        print(f"User {user.id} has registered.")

    # можем тут, например, отправлять письмо для верификации, в библиотеке всё это есть.
    # async def on_after_forgot_password(
    #     self, user: User, token: str, request: Optional[Request] = None
    # ):
    #     print(f"User {user.id} has forgot their password. Reset token: {token}")
    #
    # async def on_after_request_verify(
    #     self, user: User, token: str, request: Optional[Request] = None
    # ):
    #     print(f"Verification requested for user {user.id}. Verification token: {token}")
    #


async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)
