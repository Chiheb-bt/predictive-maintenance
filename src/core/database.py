"""
Enterprise Persistence Layer for Sentinel Industrial API.

Provides an asynchronous SQLAlchemy session and models for auditing every 
prediction and sensor reading. This signals senior-level maturity in 
data governance, auditability, and long-term ML Ops observability.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = "sqlite+aiosqlite:///sentinel.db"

# Create the engine and session factory
engine = create_async_engine(DATABASE_URL)
async_session_factory = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    pass


class PredictionAudit(Base):
    """
    Audit record for a single inference request.
    Stores raw inputs and model outputs for future drift/accuracy analysis.
    """

    __tablename__ = "prediction_audit"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(36), index=True, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)

    # Machine Data
    machine_type = Column(String(5), nullable=False)
    input_json = Column(Text, nullable=False)  # Full raw JSON payload

    # Model Results
    prediction = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)

    # Audit Metadata
    client_ip = Column(String(45), nullable=True)
    user_agent = Column(String(512), nullable=True)


async def init_db() -> None:
    """Create all tables in the sentinel.db SQLite file."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def log_prediction(
    request_id: str,
    machine_type: str,
    input_data: dict[str, Any],
    prediction: int,
    probability: float,
    risk_level: str,
    client_ip: str | None = None,
    user_agent: str | None = None,
) -> None:
    """Asynchronously log a prediction event to the audit trail."""
    async with async_session_factory() as session:
        async with session.begin():
            record = PredictionAudit(
                request_id=request_id,
                machine_type=machine_type,
                input_json=json.dumps(input_data),
                prediction=prediction,
                probability=probability,
                risk_level=risk_level,
                client_ip=client_ip,
                user_agent=user_agent,
            )
            session.add(record)
        await session.commit()


async def get_recent_audit_logs(limit: int = 10) -> list[dict[str, Any]]:
    """Retrieve the latest prediction events for the dashboard."""
    async with async_session_factory() as session:
        result = await session.execute(
            select(PredictionAudit).order_by(PredictionAudit.timestamp.desc()).limit(limit)
        )
        logs = result.scalars().all()
        return [
            {
                "id":            log.id,
                "request_id":    log.request_id,
                "timestamp":     log.timestamp.isoformat(),
                "machine_type":  log.machine_type,
                "prediction":    log.prediction,
                "probability":   round(log.probability, 4),
                "risk_level":    log.risk_level,
                "client_ip":     log.client_ip,
            }
            for log in logs
        ]
