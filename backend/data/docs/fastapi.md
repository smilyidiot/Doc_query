# FastAPI

FastAPI is a modern Python web framework for building APIs quickly. It is built
on Starlette for the web layer and Pydantic for data validation. Endpoints are
declared as plain Python functions with type hints, and FastAPI uses those type
hints to validate incoming request bodies, query parameters, and path
parameters automatically. If a request does not match the declared schema,
FastAPI returns a clear 422 validation error without any extra code.

FastAPI generates interactive OpenAPI documentation out of the box, available at
the /docs endpoint, so consumers can explore and test the API in a browser. It
supports asynchronous request handlers with async and await, dependency
injection for shared logic such as authentication, and streaming responses for
large files. Because validation and serialization are driven by Pydantic
models, request and response shapes stay consistent and self-documenting.
