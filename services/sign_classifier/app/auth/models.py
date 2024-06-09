from datetime import datetime

from fastapi_users_db_sqlalchemy import SQLAlchemyBaseUserTable
from sqlalchemy import (JSON, TIMESTAMP, Boolean, Column, ForeignKey, Integer, Float,
                        String, Table)

from .database import Base, metadata

rating = Table(
    "rating",
    metadata,
    Column("user_id", Integer, primary_key=True),
    Column("rating", Integer, nullable=False),
    Column("timestamp", TIMESTAMP, default=datetime.utcnow, nullable=False)
)

role = Table(
    "role",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("permissions", JSON),
)

results = Table(
    "results",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id",Integer, ForeignKey("rating.user_id"), nullable=False),
    Column("source_type", String, nullable=False),
    Column("result_filepath", String, nullable=False),
    Column("frame_num", Integer, nullable=True),
    Column("detection_id", Integer, nullable=False),
    Column("detection_conf", Float, nullable=False),
    Column("classification_class", Integer, nullable=False),
    Column("bbox", String, nullable=False),
    Column("frame_number", Integer, nullable=False),
    Column("detection_speed", Integer, nullable=False),

)

class User(SQLAlchemyBaseUserTable[int], Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    email = Column(String, nullable=False)
    username = Column(String, nullable=False)
    registered_at = Column(TIMESTAMP, default=datetime.utcnow)
    role_id = Column(Integer, ForeignKey(role.c.id))
    hashed_password: str = Column(String(length=1024), nullable=False)
    is_active: bool = Column(Boolean, default=True, nullable=False)
    is_superuser: bool = Column(Boolean, default=False, nullable=False)
    is_verified: bool = Column(Boolean, default=False, nullable=False)
