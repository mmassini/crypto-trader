from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone

Base = declarative_base()


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    direction = Column(String, nullable=False)   # LONG / SHORT
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    exit_price = Column(Float)
    pnl = Column(Float)
    confidence = Column(Float)
    order_id = Column(String)
    opened_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    closed_at = Column(DateTime)
    is_open = Column(Boolean, default=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    version = Column(String, nullable=False)
    sharpe = Column(Float)
    accuracy = Column(Float)
    is_champion = Column(Boolean, default=False)
    trained_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class EquitySnapshot(Base):
    __tablename__ = "equity_snapshots"

    id = Column(Integer, primary_key=True)
    balance = Column(Float)
    recorded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


engine = create_engine("sqlite:///data/trading.db")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
