# Testing Python with pytest

pytest is the most popular testing framework for Python. Tests are ordinary
functions whose names begin with test_, and assertions use the plain assert
statement; pytest rewrites assertions to give rich failure messages. Fixtures,
declared with the pytest.fixture decorator, provide reusable setup such as a
sample dataset or a configured client, and their scope can be function, module,
or session to control how often they run.

For web APIs built with FastAPI, the TestClient wraps the application so tests
can send requests in-process without starting a server. Good test suites cover
the happy path, validation errors, and edge cases such as empty input. Running
pytest with the -q flag gives quiet output, and a continuous-integration
pipeline typically runs the suite on every push to catch regressions early.
