import pytest
from utils.db import engine, Base, get_dynamic_table, create_table_if_not_exists, list_tables, insert_data, query_table

def test_dynamic_table_creation():
    """Test that a dynamic table class can be created and table exists"""
    TestTable = get_dynamic_table(Base, 'test_equities_table')
    create_table_if_not_exists(TestTable)
    tables = list_tables()
    assert 'test_equities_table' in tables

def test_insert_and_query():
    """Test inserting and querying data"""
    TestTable = get_dynamic_table(Base, 'test_equities_table')
    create_table_if_not_exists(TestTable)
    test_row = {'ticker': 'AAPL', 'date': '2025-01-01', 'open': 100, 'high': 110, 'low': 95, 'close': 105, 'volume': 1000}
    insert_data(TestTable, test_row)
    results = query_table(TestTable, filters={'ticker': 'AAPL'})
    assert len(results) > 0
    assert results[0].ticker == 'AAPL'
