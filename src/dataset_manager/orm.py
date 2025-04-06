from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from .models import *

def init_db(path="datasets/curated.sqlite") -> Session:
    db_url = "sqlite:///" + path
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    Base.metadata.create_all(engine)

    return session

def row2dict(row):
    d = {}
    for column in row.__table__.columns:
        d[column.name] = getattr(row, column.name)

    return d

def rows2dict(rows: list):
    return [row2dict(row) for row in rows]
