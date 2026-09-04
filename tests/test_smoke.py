"""Confirms the package imports and the FastAPI app is constructed."""

from app.main import app


def test_app_exists():
    assert app.title == "Cleanomatics Support Assistant"
