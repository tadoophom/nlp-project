"""SQLite database for storing annotation feedback."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func


DB_PATH = Path(__file__).parent.parent / "data" / "feedback.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    keyword = Column(String, nullable=False)
    sentence = Column(String, nullable=False)
    classification = Column(String, nullable=False)
    correct_label = Column(Boolean, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def insert_feedback(
    keyword: str,
    sentence: str,
    classification: str,
    correct_label: bool,
) -> None:
    session = SessionLocal()
    try:
        rec = Feedback(
            keyword=keyword,
            sentence=sentence,
            classification=classification,
            correct_label=correct_label,
        )
        session.add(rec)
        session.commit()
    finally:
        session.close()


def get_feedback_summary(limit: int = 20):
    session = SessionLocal()
    try:
        agg = (
            session.query(
                Feedback.keyword,
                Feedback.classification,
                func.sum(func.case((Feedback.correct_label == True, 1), else_=0)).label("correct"),
                func.sum(func.case((Feedback.correct_label == False, 1), else_=0)).label("incorrect"),
            )
            .group_by(Feedback.keyword, Feedback.classification)
            .all()
        )
        recent = (
            session.query(Feedback)
            .order_by(Feedback.created_at.desc())
            .limit(limit)
            .all()
        )
        return agg, recent
    finally:
        session.close()