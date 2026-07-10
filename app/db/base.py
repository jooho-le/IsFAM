from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for Postgres-backed domains (auth, family, devices, ...)."""
