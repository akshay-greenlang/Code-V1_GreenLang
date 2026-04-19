# -*- coding: utf-8 -*-
"""
Network guard utilities to ensure tests run offline.
"""
import socket
import urllib.request
import urllib.error
from unittest.mock import patch, MagicMock
from contextlib import contextmanager
from typing import Optional, Callable, Any


class NetworkGuard:
    """
    Guard against network access in tests.
    
    Blocks all socket connections and HTTP requests to ensure
    tests are deterministic and don't depend on external services.
    """
    
    def __init__(self, allow_localhost: bool = False):
        """
        Initialize network guard.
        
        Args:
            allow_localhost: If True, allow connections to localhost/127.0.0.1
        """
        self.allow_localhost = allow_localhost
        self._original_socket = socket.socket
        self._original_create_connection = socket.create_connection
        self._original_urlopen = urllib.request.urlopen
        self._patches = []
    
    def _is_localhost(self, host: str) -> bool:
        """Check if host is localhost."""
        return host in ('localhost', '127.0.0.1', '::1', '0.0.0.0')
    
    def _block_socket(self, *args, **kwargs):
        """Block socket creation."""
        # Check if this is a localhost connection
        if self.allow_localhost and args and len(args) > 1:
            address = args[1] if len(args) > 1 else kwargs.get('address', ())
            if address and self._is_localhost(address[0]):
                return self._original_socket(*args, **kwargs)
        
        raise NetworkAccessError(
            "Network access blocked in test. "
            "All external connections must be mocked."
        )
    
    def _block_connection(self, address, *args, **kwargs):
        """Block connection creation."""
        if self.allow_localhost and self._is_localhost(address[0]):
            return self._original_create_connection(address, *args, **kwargs)
        
        raise NetworkAccessError(
            f"Network connection to {address[0]}:{address[1]} blocked. "
            "Use mocks for external services."
        )
    
    def _block_urlopen(self, url, *args, **kwargs):
        """Block URL opening."""
        url_str = str(url)
        if self.allow_localhost and any(
            local in url_str for local in ['localhost', '127.0.0.1']
        ):
            return self._original_urlopen(url, *args, **kwargs)
        
        raise NetworkAccessError(
            f"HTTP request to {url_str} blocked. "
            "Use mocks for external APIs."
        )
    
    def enable(self):
        """Enable network blocking."""
        socket.socket = self._block_socket
        socket.create_connection = self._block_connection
        urllib.request.urlopen = self._block_urlopen
        
        # Also patch common HTTP libraries if available
        try:
            import requests
            self._patches.append(
                patch.object(requests, 'get', side_effect=self._block_request)
            )
            self._patches.append(
                patch.object(requests, 'post', side_effect=self._block_request)
            )
            self._patches.append(
                patch.object(requests, 'put', side_effect=self._block_request)
            )
            self._patches.append(
                patch.object(requests, 'delete', side_effect=self._block_request)
            )
            for p in self._patches:
                p.start()
        except ImportError:
            pass
        
        # Patch httpx if available
        try:
            import httpx
            self._patches.append(
                patch.object(httpx, 'get', side_effect=self._block_request)
            )
            self._patches.append(
                patch.object(httpx, 'post', side_effect=self._block_request)
            )
            for p in self._patches[-2:]:
                p.start()
        except ImportError:
            pass
    
    def _block_request(self, *args, **kwargs):
        """Block HTTP request."""
        url = args[0] if args else kwargs.get('url', '')
        raise NetworkAccessError(
            f"HTTP request to {url} blocked. Use mocks for external APIs."
        )
    
    def disable(self):
        """Disable network blocking and restore original functions."""
        socket.socket = self._original_socket
        socket.create_connection = self._original_create_connection
        urllib.request.urlopen = self._original_urlopen
        
        # Stop all patches
        for p in self._patches:
            p.stop()
        self._patches.clear()


class NetworkAccessError(Exception):
    """Raised when network access is attempted in tests."""
    pass


@contextmanager
def block_network(allow_localhost: bool = False):
    """
    Context manager to block network access.
    
    Usage:
        with block_network():
            # Network access blocked here
            run_test()
    
    Args:
        allow_localhost: If True, allow localhost connections
    """
    guard = NetworkGuard(allow_localhost=allow_localhost)
    guard.enable()
    try:
        yield guard
    finally:
        guard.disable()


def mock_http_response(status_code: int = 200, 
                      json_data: Optional[dict] = None,
                      text: Optional[str] = None,
                      headers: Optional[dict] = None):
    """
    Create a mock HTTP response.
    
    Args:
        status_code: HTTP status code
        json_data: JSON response data
        text: Text response data
        headers: Response headers
    
    Returns:
        Mock response object
    """
    response = MagicMock()
    response.status_code = status_code
    response.ok = 200 <= status_code < 300
    response.headers = headers or {}
    
    if json_data is not None:
        response.json.return_value = json_data
        response.text = json.dumps(json_data)
    elif text is not None:
        response.text = text
        response.json.side_effect = ValueError("No JSON data")
    else:
        response.text = ""
        response.json.side_effect = ValueError("No JSON data")
    
    return response


class APICallRecorder:
    """Record API calls for verification."""
    
    def __init__(self):
        self.calls = []
    
    def record(self, method: str, url: str, **kwargs):
        """Record an API call."""
        self.calls.append({
            'method': method,
            'url': url,
            'kwargs': kwargs
        })
    
    def assert_called(self, method: str, url: str, times: int = 1):
        """Assert that a specific API call was made."""
        matching_calls = [
            call for call in self.calls
            if call['method'] == method and call['url'] == url
        ]
        
        if len(matching_calls) != times:
            raise AssertionError(
                f"Expected {times} calls to {method} {url}, "
                f"but found {len(matching_calls)}"
            )
    
    def assert_not_called(self, method: str = None, url: str = None):
        """Assert that no API calls were made (or specific ones)."""
        if method is None and url is None:
            if self.calls:
                raise AssertionError(
                    f"Expected no API calls, but found {len(self.calls)}"
                )
        else:
            matching_calls = [
                call for call in self.calls
                if (method is None or call['method'] == method) and
                   (url is None or call['url'] == url)
            ]
            if matching_calls:
                raise AssertionError(
                    f"Expected no calls to {method or 'any'} {url or 'any URL'}, "
                    f"but found {len(matching_calls)}"
                )
    
    def clear(self):
        """Clear recorded calls."""
        self.calls.clear()


@contextmanager
def mock_api_endpoint(url_pattern: str, 
                     response_func: Callable[[str, Any], dict],
                     method: str = 'GET'):
    """
    Mock a specific API endpoint.
    
    Args:
        url_pattern: URL pattern to match (can include wildcards)
        response_func: Function that returns response data
        method: HTTP method to mock
    
    Usage:
        def emission_api(url, data):
            return {'emissions': 1000.0}
        
        with mock_api_endpoint('/api/emissions/*', emission_api):
            # API calls to /api/emissions/* will use emission_api
            result = calculate_emissions()
    """
    import re
    from unittest.mock import patch
    
    # Convert pattern to regex
    pattern = re.escape(url_pattern).replace(r'\*', '.*')
    pattern_re = re.compile(pattern)
    
    def mock_request(url, *args, **kwargs):
        if pattern_re.match(url):
            data = kwargs.get('json') or kwargs.get('data')
            response_data = response_func(url, data)
            return mock_http_response(json_data=response_data)
        else:
            raise NetworkAccessError(f"Unmocked API call to {url}")
    
    try:
        import requests
        method_lower = method.lower()
        with patch.object(requests, method_lower, side_effect=mock_request):
            yield
    except ImportError:
        # If requests not available, just yield
        yield