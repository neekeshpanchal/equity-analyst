# utils/db.py
from sqlalchemy import create_engine, Column, Integer, Float, String, Date, MetaData, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, registry
from sqlalchemy.exc import IntegrityError
from datetime import datetime, date
import pandas as pd
from config import DATABASE_URI, EQUITIES_SUFFIX, METRICS_SUFFIX, BACKTEST_SUFFIX, PREDICTIONS_SUFFIX

# -------------------------------
# DB Initialization
# -------------------------------
engine = create_engine(DATABASE_URI, echo=False, future=True)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()
mapper_registry = registry()  # SQLAlchemy 2.0 classical mapping

# -------------------------------
# Base Classes
# -------------------------------
class BaseEquities(Base):
    __tablename__ = 'equities_data'
    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

class BaseMetrics(Base):
    __tablename__ = 'equities_metrics'
    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    avg = Column(Float)
    max = Column(Float)
    min = Column(Float)
    volatility = Column(Float)
    sharpe = Column(Float)
    drawdown = Column(Float)

class BaseBacktests(Base):
    __tablename__ = 'backtests'
    id = Column(Integer, primary_key=True)
    strategy = Column(String, nullable=False)
    ticker = Column(String, nullable=False)
    start_date = Column(Date)
    end_date = Column(Date)
    return_pct = Column(Float)
    volatility = Column(Float)
    sharpe = Column(Float)
    drawdown = Column(Float)

class BasePredictions(Base):
    __tablename__ = 'ml_predictions'
    id = Column(Integer, primary_key=True)
    model_name = Column(String, nullable=False)
    ticker = Column(String, nullable=False)
    date = Column(Date)
    predicted_price = Column(Float)

# -------------------------------
# Dynamic Table Handling
# -------------------------------
def get_dynamic_table(base_class, table_name: str, extra_columns: dict = None):
    """
    Create a standalone table dynamically based on base_class schema.
    Uses SQLAlchemy 2.0 registry for classical mapping.
    """
    extra_columns = extra_columns or {}

    # Copy base columns
    base_columns = {c.name: c.copy() for c in base_class.__table__.columns}

    # Convert extra columns (passed as SQLAlchemy types) to Column objects
    for col_name, col_type in extra_columns.items():
        if not isinstance(col_type, Column):
            base_columns[col_name] = Column(col_name, col_type)

    # Create new Table object
    metadata = base_class.metadata
    new_table = Table(table_name, metadata, *(base_columns.values()), extend_existing=True)

    # Map a dynamic class
    class DynamicTable:
        pass

    mapper_registry.map_imperatively(DynamicTable, new_table)
    return DynamicTable

def create_table_if_not_exists(table_class):
    """Create table in DB if it doesn't exist"""
    table_class.__table__.create(bind=engine, checkfirst=True)

def list_tables():
    """Return list of all tables in DB"""
    meta = MetaData()
    meta.reflect(bind=engine)
    return list(meta.tables.keys())

# -------------------------------
# Data Operations
# -------------------------------
def clean_record_dates(record: dict):
    """Convert pandas Timestamps to datetime for SQLite compatibility"""
    new_record = {}
    for k, v in record.items():
        if isinstance(v, pd.Timestamp):
            new_record[k] = v.to_pydatetime()
        elif isinstance(v, (datetime, date)):
            new_record[k] = v
        else:
            new_record[k] = v
    return new_record

def insert_data(table_class, data_dict):
    """Insert single row dict into table"""
    data_dict = clean_record_dates(data_dict)
    obj = table_class(**data_dict)
    try:
        session.add(obj)
        session.commit()
    except IntegrityError:
        session.rollback()
        raise

def bulk_insert(table_class, data_list):
    """Insert multiple rows at once, handling timestamps"""
    cleaned_data = [clean_record_dates(d) for d in data_list]
    objs = [table_class(**d) for d in cleaned_data]
    try:
        session.bulk_save_objects(objs)
        session.commit()
    except IntegrityError:
        session.rollback()
        raise

def query_table(table_class, filters=None, limit=None):
    """Query table with optional filters"""
    q = session.query(table_class)
    if filters:
        for attr, value in filters.items():
            if hasattr(table_class, attr):
                q = q.filter(getattr(table_class, attr) == value)
    if limit:
        q = q.limit(limit)
    return q.all()

# -------------------------------
# Utility for Dropdown Options
# -------------------------------
def get_table_options(base_class_type):
    tables = list_tables()
    filtered = []
    for table_name in tables:
        if base_class_type == 'metrics' and table_name.endswith(METRICS_SUFFIX):
            filtered.append(table_name)
        elif base_class_type == 'equities' and table_name.endswith(EQUITIES_SUFFIX):
            filtered.append(table_name)
        elif base_class_type == 'backtests' and table_name.endswith(BACKTEST_SUFFIX):
            filtered.append(table_name)
        elif base_class_type == 'predictions' and table_name.endswith(PREDICTIONS_SUFFIX):
            filtered.append(table_name)
    return filtered
