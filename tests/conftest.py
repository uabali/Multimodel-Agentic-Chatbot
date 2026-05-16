"""Shared test environment defaults."""

import os


os.environ.setdefault("APP_ADMIN_PASSWORD", "test-admin-password")
os.environ.setdefault("APP_PASSWORD_SALT", "test-password-salt")
os.environ.setdefault("CHAINLIT_AUTH_SECRET", "test-chainlit-auth-secret")
