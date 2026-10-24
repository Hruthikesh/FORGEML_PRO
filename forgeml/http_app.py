"""Explicit Starlette/Pydantic fallback when offline FastAPI wheels are unavailable.

This is a small route adapter for ForgeML, not a replacement FastAPI package.
It intentionally does not expose OpenAPI/docs or claim FastAPI is installed.
"""

from __future__ import annotations

import inspect
import json
from typing import get_type_hints

from pydantic import BaseModel, ValidationError
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route


class StarletteApplication(Starlette):
    def __init__(self, **metadata):
        super().__init__()
        self.state.framework = "Starlette + Pydantic (offline fallback)"
        self.state.metadata = metadata

    def middleware(self, kind):
        if kind != "http":
            raise ValueError("Only HTTP middleware is supported")

        def decorator(function):
            from starlette.middleware.base import BaseHTTPMiddleware

            self.add_middleware(BaseHTTPMiddleware, dispatch=function)
            return function

        return decorator

    def _register(self, path, method, response_class=None, **options):
        def decorator(function):
            signature = inspect.signature(function)
            annotations = get_type_hints(function)

            async def endpoint(request):
                parameters = {}
                for name, parameter in signature.parameters.items():
                    annotation = annotations.get(name, parameter.annotation)
                    if name in request.path_params:
                        parameters[name] = request.path_params[name]
                    elif annotation is Request:
                        parameters[name] = request
                    elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
                        try:
                            body = await request.json()
                            parameters[name] = annotation.model_validate(body)
                        except ValidationError as error:
                            return JSONResponse(
                                {"detail": error.errors(include_url=False, include_context=False)}, status_code=422
                            )
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            return JSONResponse({"detail": "Request body must be valid JSON"}, status_code=422)
                    elif parameter.default is inspect.Parameter.empty:
                        raise HTTPException(400, "Missing request parameter")
                try:
                    result = await run_in_threadpool(function, **parameters)
                    if isinstance(result, Response):
                        return result
                    if response_class is not None:
                        return response_class(result)
                    return JSONResponse(result)
                except HTTPException as error:
                    return JSONResponse({"detail": error.detail}, status_code=error.status_code)

            endpoint.sync_handler = function
            self.router.routes.append(Route(path, endpoint, methods=[method], name=function.__name__))
            return function

        return decorator

    def get(self, path, **options):
        return self._register(path, "GET", **options)

    def post(self, path, **options):
        return self._register(path, "POST", **options)
