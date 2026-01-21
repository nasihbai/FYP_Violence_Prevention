"""
Web Dashboard Package
=====================
Flask-based web interface for violence detection monitoring.
"""

from .app import app, create_app, run_server, initialize_detector

__all__ = ['app', 'create_app', 'run_server', 'initialize_detector']
