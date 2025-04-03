from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Float, func
from sqlalchemy.orm import Session
from sqlalchemy.orm import relationship, declarative_base, sessionmaker
import pprint

Base = declarative_base()

# Define ORM Models
class ArticleRelevance(Base):
    __tablename__ = "article_relevance"
    article_id = Column(Integer, ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True)
    statement_id = Column(Integer, ForeignKey("statements.id", ondelete="CASCADE"), primary_key=True)

class Article(Base):
    __tablename__ = "articles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(Text)
    title = Column(Text)
    description = Column(Text)
    content = Column(Text)
    type = Column(Text)
    author = Column(Text)
    source = Column(Text)
    published = Column(Text)
    accessed = Column(Text)
    
    statements = relationship("Statement", secondary="article_relevance", back_populates="articles")
    segments = relationship("Segment", back_populates="article")

class EvidenceCheat(Base):
    __tablename__ = "evidence_cheat"
    id = Column(Integer, primary_key=True, autoincrement=True)
    statement_id = Column(Integer, ForeignKey("statements.id"))
    title = Column(Text, default='Článek')
    content = Column(Text)
    relevant_paragraph = Column(Text)
    url = Column(Text)
    relevance = Column(Float, default=1.0)

class SegmentRelevance(Base):
    __tablename__ = "segment_relevance"
    segment_id = Column(Integer, ForeignKey("segments.id", ondelete="CASCADE"), primary_key=True)
    statement_id = Column(Integer, ForeignKey("statements.id", ondelete="CASCADE"), primary_key=True)

    relevance = Column(Float, default=1.0)

class Segment(Base):
    __tablename__ = "segments"
    id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(Integer, ForeignKey("articles.id"))
    text = Column(Text)
    
    article = relationship("Article", back_populates="segments")
    statements = relationship("Statement", secondary="segment_relevance", back_populates="segments")

class Statement(Base):
    __tablename__ = "statements"
    id = Column(Integer, primary_key=True, autoincrement=True)
    statement = Column(Text)
    label = Column(Text)
    author = Column(Text)
    date = Column(Text)
    party = Column(Text)
    explanation = Column(Text)
    explanation_brief = Column(Text)
    origin = Column(Text)
    
    articles = relationship("Article", secondary="article_relevance", back_populates="statements")
    segments = relationship("Segment", secondary="segment_relevance", back_populates="statements")
    tags = relationship("Tag", back_populates="statement")

class Tag(Base):
    __tablename__ = "tags"
    id = Column(Integer, primary_key=True, autoincrement=True)
    statement_id = Column(Integer, ForeignKey("statements.id"))
    tag = Column(Text)
    
    statement = relationship("Statement", back_populates="tags")

def as_dict(row) -> dict:
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}

def as_dict_list(rows) -> list[dict]:
    return [as_dict(row) for row in rows]

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
        d[column.name] = str(getattr(row, column.name))

    return d

def rows2dict(rows: list):
    return [row2dict(row) for row in rows]

if __name__ == "__main__":
    dataset = init_db("../../datasets/curated_updated.sqlite")
    statements_with_segments = (
        dataset.query(Statement)
        .join(Statement.segments)  # Join through the segments relationship
        .join(SegmentRelevance)  # Join the segment_relevance table to filter by relevance
        .filter(SegmentRelevance.relevance == 0.5)  # Filter for relevance 1
        .all()  # Get all the results
    )

    for statement in statements_with_segments:
        print([seg.text for seg in statement.segments])
