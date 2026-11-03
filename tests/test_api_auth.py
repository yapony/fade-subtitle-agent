import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from fade_content_agent.api import app, require_api_key


class ApiAuthTests(unittest.TestCase):
    def test_missing_server_key_fails_closed(self):
        with patch.dict(os.environ, {"FADE_AGENT_API_KEY": ""}):
            with self.assertRaises(HTTPException) as raised:
                require_api_key("Bearer secret")
        self.assertEqual(raised.exception.status_code, 503)

    def test_missing_or_invalid_bearer_key_is_rejected(self):
        with patch.dict(os.environ, {"FADE_AGENT_API_KEY": "secret"}):
            for authorization in (None, "Basic secret", "Bearer wrong"):
                with self.subTest(authorization=authorization):
                    with self.assertRaises(HTTPException) as raised:
                        require_api_key(authorization)
                    self.assertEqual(raised.exception.status_code, 401)

    def test_valid_bearer_key_is_accepted(self):
        with patch.dict(os.environ, {"FADE_AGENT_API_KEY": "secret"}):
            self.assertIsNone(require_api_key("Bearer secret"))

    def test_analysis_routes_require_auth_but_health_does_not(self):
        routes = {route.path: route for route in app.routes if hasattr(route, "dependant")}
        for path in ("/analyses", "/analyses/{analysis_id}", "/analyses/{analysis_id}/result"):
            with self.subTest(path=path):
                calls = {dependency.call for dependency in routes[path].dependant.dependencies}
                self.assertIn(require_api_key, calls)
        health = routes["/health"]
        self.assertNotIn(require_api_key, {dependency.call for dependency in health.dependant.dependencies})


if __name__ == "__main__":
    unittest.main()
