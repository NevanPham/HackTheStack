"""
Pytest tests for strict Lab Mode gating.

Tests verify that Lab Mode is properly gated based on environment variables
and request headers, ensuring vulnerabilities are only accessible when explicitly enabled.
"""

import pytest
import os
from fastapi.testclient import TestClient
from backend.main import app
from backend.settings import get_settings, resolve_mode
from starlette.requests import Request


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_request():
    """Create a mock request object for testing resolve_mode."""
    from starlette.requests import Request
    from starlette.datastructures import Headers
    
    class MockRequest:
        def __init__(self, headers=None):
            # Convert dict to Headers object to match FastAPI's Request
            if headers:
                self.headers = Headers(headers)
            else:
                self.headers = Headers({})
    
    return MockRequest


class TestProductionDisablesLab:
    """Test A: Production environment disables lab mode no matter what."""
    
    def test_production_with_lab_enabled_header(self, client, monkeypatch, mock_request):
        """Test that production environment blocks lab mode even with header."""
        # Set environment to production with LAB_MODE_ENABLED=true
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("LAB_MODE_ENABLED", "true")
        
        # Reload settings by clearing any cached values
        # Note: In a real scenario, you might need to reload the module
        # For this test, we'll call get_settings() which reads env at runtime
        
        # Make request with X-Lab-Mode header
        response = client.get("/api/mode", headers={"X-Lab-Mode": "true"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "secure"
        assert data["lab_available"] is False
    
    def test_production_with_lab_enabled_no_header(self, client, monkeypatch):
        """Test that production environment blocks lab mode without header."""
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("LAB_MODE_ENABLED", "true")
        
        response = client.get("/api/mode")
        
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "secure"
        assert data["lab_available"] is False


class TestDevDisabledBlocksLab:
    """Test B: Development environment with LAB_MODE_ENABLED=false blocks lab."""
    
    def test_dev_disabled_with_header(self, client, monkeypatch):
        """Test that dev with LAB_MODE_ENABLED=false blocks lab even with header."""
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("LAB_MODE_ENABLED", "false")
        
        response = client.get("/api/mode", headers={"X-Lab-Mode": "true"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "secure"
        assert data["lab_available"] is False
    
    def test_dev_disabled_no_header(self, client, monkeypatch):
        """Test that dev with LAB_MODE_ENABLED=false blocks lab without header."""
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("LAB_MODE_ENABLED", "false")
        
        response = client.get("/api/mode")
        
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "secure"
        assert data["lab_available"] is False


class TestDevEnabledAllowsLab:
    """Test C: Development environment with LAB_MODE_ENABLED=true allows lab."""
    
    def test_dev_enabled_with_header(self, client, monkeypatch):
        """Test that dev with LAB_MODE_ENABLED=true allows lab with header."""
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("LAB_MODE_ENABLED", "true")
        
        response = client.get("/api/mode", headers={"X-Lab-Mode": "true"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "lab"
        assert data["lab_available"] is True
    
    def test_dev_enabled_with_header_case_insensitive(self, client, monkeypatch):
        """Test that header is case-insensitive."""
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("LAB_MODE_ENABLED", "true")
        
        # Test various case combinations
        for header_value in ["True", "TRUE", "true", "1", "yes", "YES"]:
            response = client.get("/api/mode", headers={"X-Lab-Mode": header_value})
            assert response.status_code == 200
            data = response.json()
            assert data["mode"] == "lab", f"Failed for header value: {header_value}"
            assert data["lab_available"] is True


class TestDefaultSecureWithoutHeader:
    """Test D: Default secure mode without header."""
    
    def test_dev_enabled_no_header_defaults_secure(self, client, monkeypatch):
        """Test that dev with LAB_MODE_ENABLED=true defaults to secure without header."""
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("LAB_MODE_ENABLED", "true")
        
        response = client.get("/api/mode")
        
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "secure"
        assert data["lab_available"] is True  # Available but not active
    
    def test_dev_enabled_invalid_header_defaults_secure(self, client, monkeypatch):
        """Test that invalid header values default to secure."""
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("LAB_MODE_ENABLED", "true")
        
        # Test invalid header values
        for header_value in ["false", "0", "no", "invalid", ""]:
            response = client.get("/api/mode", headers={"X-Lab-Mode": header_value})
            assert response.status_code == 200
            data = response.json()
            assert data["mode"] == "secure", f"Failed for header value: {header_value}"
            assert data["lab_available"] is True


class TestResolveModeFunction:
    """Test the resolve_mode function directly."""
    
    def test_resolve_mode_production_always_secure(self, monkeypatch, mock_request):
        """Test that resolve_mode returns secure in production."""
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("LAB_MODE_ENABLED", "true")
        
        # Create request with lab mode header
        request = mock_request(headers={"X-Lab-Mode": "true"})
        
        # Need to reload settings module to pick up new env vars
        # For testing, we'll import and call directly
        from backend.settings import resolve_mode
        mode = resolve_mode(request)
        
        assert mode == "secure"
    
    def test_resolve_mode_dev_disabled_always_secure(self, monkeypatch, mock_request):
        """Test that resolve_mode returns secure when lab is disabled."""
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("LAB_MODE_ENABLED", "false")
        
        request = mock_request(headers={"X-Lab-Mode": "true"})
        
        from backend.settings import resolve_mode
        mode = resolve_mode(request)
        
        assert mode == "secure"
    
    def test_resolve_mode_dev_enabled_with_header(self, monkeypatch, mock_request):
        """Test that resolve_mode returns lab when enabled and header present."""
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("LAB_MODE_ENABLED", "true")
        
        request = mock_request(headers={"X-Lab-Mode": "true"})
        
        from backend.settings import resolve_mode
        mode = resolve_mode(request)
        
        assert mode == "lab"
    
    def test_resolve_mode_dev_enabled_no_header(self, monkeypatch, mock_request):
        """Test that resolve_mode returns secure without header."""
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("LAB_MODE_ENABLED", "true")
        
        request = mock_request(headers={})
        
        from backend.settings import resolve_mode
        mode = resolve_mode(request)
        
        assert mode == "secure"


class TestLabOnlyEndpoints:
    """Test that lab-only endpoints are properly gated."""
    
    def test_debug_info_blocked_in_production(self, client, monkeypatch):
        """Test that /debug/info is blocked in production."""
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("LAB_MODE_ENABLED", "true")
        
        response = client.get("/debug/info", headers={"X-Lab-Mode": "true"})
        
        assert response.status_code == 403
    
    def test_debug_info_blocked_when_disabled(self, client, monkeypatch):
        """Test that /debug/info is blocked when lab is disabled."""
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("LAB_MODE_ENABLED", "false")
        
        response = client.get("/debug/info", headers={"X-Lab-Mode": "true"})
        
        assert response.status_code == 403
    
    def test_debug_info_allowed_when_enabled(self, client, monkeypatch):
        """Test that /debug/info is allowed when lab is enabled."""
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("LAB_MODE_ENABLED", "true")
        
        response = client.get("/debug/info", headers={"X-Lab-Mode": "true"})
        
        assert response.status_code == 200
    
    def test_debug_health_blocked_in_production(self, client, monkeypatch):
        """Test that /debug/health is blocked in production."""
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("LAB_MODE_ENABLED", "true")
        
        response = client.get("/debug/health", headers={"X-Lab-Mode": "true"})
        
        assert response.status_code == 403

