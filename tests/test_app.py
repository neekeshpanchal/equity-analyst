# tests/test_app.py
import pytest
from app import app
from utils.db import list_tables, get_dynamic_table, create_table_if_not_exists
from utils.db import session, BaseEquities

@pytest.fixture
def client():
    """Flask test client fixture"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# -------------------------------
# Data Dashboard Tests
# -------------------------------

def test_data_dashboard_get(client):
    """GET request should return 200 and contain dashboard form"""
    response = client.get('/data_dashboard')
    assert response.status_code == 200
    assert b"Select Table" in response.data or b"table" in response.data

def test_data_dashboard_post(client):
    """POST request inserts data into a test table"""
    # Create temporary test table
    test_table_name = 'test_equities_data'
    TableClass = get_dynamic_table(BaseEquities, test_table_name)
    create_table_if_not_exists(TableClass)

    payload = {
        'ticker': 'AAPL',
        'start': '2025-01-01',
        'end': '2025-01-10',
        'table': test_table_name
    }

    response = client.post('/data_dashboard', data=payload, follow_redirects=True)
    assert response.status_code == 200
    assert b"saved to" in response.data or b"Error" not in response.data

    # Ensure data inserted in DB
    results = session.query(TableClass).all()
    assert len(results) > 0

# -------------------------------
# Backtest Dashboard Tests
# -------------------------------

def test_backtest_dashboard_get(client):
    """GET request returns backtest dashboard"""
    response = client.get('/backtest_dashboard')
    assert response.status_code == 200
    assert b"Backtest" in response.data or b"table" in response.data

def test_backtest_dashboard_post_invalid(client):
    """POST with missing data returns error in result"""
    payload = {
        'ticker': 'AAPL',
        'table': 'nonexistent_table',
        'start': '2025-01-01',
        'end': '2025-01-10',
        'model': 'lstm_test'
    }
    response = client.post('/backtest_dashboard', data=payload, follow_redirects=True)
    assert response.status_code == 200
    assert b"No data available" in response.data or b"error" in response.data.lower()

# -------------------------------
# API Endpoint Tests
# -------------------------------

def test_api_fetch_metrics(client):
    """Test POST /api/fetch_metrics returns success"""
    test_table_name = 'test_api_table'
    TableClass = get_dynamic_table(BaseEquities, test_table_name)
    create_table_if_not_exists(TableClass)

    payload = {
        'ticker': 'MSFT',
        'start': '2025-01-01',
        'end': '2025-01-05',
        'table': test_table_name
    }

    response = client.post('/api/fetch_metrics', json=payload)
    json_data = response.get_json()
    assert response.status_code == 200
    assert json_data['status'] == 'success'
    assert json_data['rows'] > 0
