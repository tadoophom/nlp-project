"""
database.py – simple SQLite database for storing annotation feedback.

This module defines a tiny data access layer around a single ``feedback``
table. Each row records a keyword hit emitted by the NLP pipeline and
captures whether a user indicated that the polarity classification was
correct or incorrect via the Streamlit UI. Capturing feedback enables
active learning workflows where mislabeled examples can be used to
improve future models.

The database is initialised lazily on first use and persists in the
``feedback.db`` file in the project root. For production deployments
SQLAlchemy can be configured to connect to PostgreSQL or another RDBMS
via environment variables.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

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


DB_PATH = Path(__file__).parent / "feedback.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Feedback(Base):  # type: ignore[misc]
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    keyword = Column(String, nullable=False)
    sentence = Column(String, nullable=False)
    classification = Column(String, nullable=False)
    correct_label = Column(Boolean, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    """Create tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)


def insert_feedback(
    keyword: str,
    sentence: str,
    classification: str,
    correct_label: bool,
) -> None:
    """
    Persist a single feedback record. In case of failure, the exception
    propagates to the caller for Streamlit to handle gracefully.
    """
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
    """Return aggregate counts and recent feedback rows for dashboarding."""
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