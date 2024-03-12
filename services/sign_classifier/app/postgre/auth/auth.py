from fastapi_users.authentication import AuthenticationBackend, CookieTransport
from fastapi_users.authentication import JWTStrategy

cookie_transport = CookieTransport(cookie_name="finance", cookie_max_age=3600)

# секрет который кодирует-декодирует токены, должен храниться максимально надежно;
# примеры как это работает можно посмотреть в https://jwt.io/
SECRET = "SECRET"


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)


auth_backend = AuthenticationBackend(
    name="jwt",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)