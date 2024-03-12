from datetime import datetime

from sqlalchemy import MetaData, Table, Column, Integer, String, TIMESTAMP, ForeignKey, JSON, Boolean

metadata = MetaData()

rating = Table(
    "rating",
    metadata,
    Column("user_id", Integer, primary_key=True),
    Column("rating", Integer, nullable=False),
    Column("timestamp", TIMESTAMP, default=datetime.utcnow, nullable=False)
)

user = Table(
    "user",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("email", String, nullable=False),
    Column("username", String, nullable=False),
    Column("hashed_password", String, nullable=False),
    Column("registered_at", TIMESTAMP, default=datetime.utcnow, nullable=False),
    Column("role_id", Integer, nullable=False),
    Column("is_active", Boolean, default=True, nullable=False),
    Column("is_superuser", Boolean, default=True, nullable=False),
)