"""Prediction audit log storage (FR9). Demo-scale: SQLAlchemy + a single
table, tables created at startup — no migration framework, this isn't a
production deployment (CLAUDE.md §2.5)."""

from __future__ import annotations

import datetime as dt
import json

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src import config


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    patient_id = Column(String, nullable=False, index=True)
    risk_score = Column(Float, nullable=False)
    alert = Column(Boolean, nullable=False)
    top_features = Column(Text, nullable=False)  # JSON-encoded list[str]
    model_name = Column(String, nullable=False)
    latency_ms = Column(Float, nullable=False)


engine = create_engine(config.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def log_prediction(session: Session, response) -> None:
    entry = PredictionLog(
        patient_id=response.patient_id,
        risk_score=response.risk_score,
        alert=response.alert,
        top_features=json.dumps(response.top_features),
        model_name=response.model_name,
        latency_ms=response.latency_ms,
    )
    session.add(entry)
    session.commit()
