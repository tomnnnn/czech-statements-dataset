from sqlalchemy import create_engine, Integer, String, Text, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship, declarative_base, sessionmaker
from sqlalchemy.orm import Session

Base = declarative_base()

class ArticleRelevance(Base):
    __tablename__ = "article_relevance"
    article_id = mapped_column(Integer, ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True)
    statement_id = mapped_column(Integer, ForeignKey("statements.id", ondelete="CASCADE"), primary_key=True)

class Article(Base):
    __tablename__ = "articles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(Text, nullable=True)
    author: Mapped[str] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(Text, nullable=True)
    published: Mapped[str] = mapped_column(Text, nullable=True)
    accessed: Mapped[str] = mapped_column(Text, nullable=True)
    
    statements = relationship("Statement", secondary="article_relevance", back_populates="articles")
    segments = relationship("Segment", back_populates="article")

class EvidenceCheat(Base):
    __tablename__ = "evidence_cheat"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    statement_id: Mapped[int] = mapped_column(Integer, ForeignKey("statements.id"))
    title: Mapped[str] = mapped_column(Text, default='Článek')
    content: Mapped[str] = mapped_column(Text)
    relevant_paragraph: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    relevance: Mapped[float] = mapped_column(Float, default=1.0)

class SegmentRelevance(Base):
    __tablename__ = "segment_relevance"
    segment_id: Mapped[int] = mapped_column(Integer, ForeignKey("segments.id", ondelete="CASCADE"), primary_key=True)
    statement_id: Mapped[int] = mapped_column(Integer, ForeignKey("statements.id", ondelete="CASCADE"), primary_key=True)
    relevance: Mapped[float] = mapped_column(Float, default=1.0)

class Segment(Base):
    __tablename__ = "segments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(Integer, ForeignKey("articles.id"))
    text: Mapped[str] = mapped_column(Text)
    
    article = relationship("Article", back_populates="segments")
    statements = relationship("Statement", secondary="segment_relevance", back_populates="segments")

class Statement(Base):
    __tablename__ = "statements"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    statement: Mapped[str] = mapped_column(Text)
    label: Mapped[str] = mapped_column(Text)
    author: Mapped[str] = mapped_column(Text)
    date: Mapped[str] = mapped_column(Text)
    party: Mapped[str] = mapped_column(Text)
    explanation: Mapped[str] = mapped_column(Text)
    explanation_brief: Mapped[str] = mapped_column(Text)
    origin: Mapped[str] = mapped_column(Text)
    
    articles = relationship("Article", secondary="article_relevance", back_populates="statements")
    segments = relationship("Segment", secondary="segment_relevance", back_populates="statements")
    tags = relationship("Tag", back_populates="statement")

class Tag(Base):
    __tablename__ = "tags"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    statement_id: Mapped[int] = mapped_column(Integer, ForeignKey("statements.id"))
    tag: Mapped[str] = mapped_column(Text)
    
    statement = relationship("Statement", back_populates="tags")

