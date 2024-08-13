import pytest
from fastapi.testclient import TestClient
# creates app = FastAPI() and logger = logging.getLogger('uvicorn.error')
import fapi

class RcMock:
    @staticmethod
    def keys(**kwargs) -> list:
        return ["data_1", "data_2"]
    @staticmethod
    def zrange(n, *args, **kwargs) -> list:
        if n == "data_1":
            return [(b"Name1", 1),  (b"Name2", 2)]
        else:
            return [(b"Name11", 11), (b"Name12", 12)]

@pytest.fixture
def client():
    return TestClient(fapi.app)


def test_get_all(client):
    fapi.rc = RcMock
    response = client.get("/names")
    assert response.status_code == 200
    assert response.json() == [{'1': 'Name1'}, {'2': 'Name2'}, {'11': 'Name11'}, {'12': 'Name12'}]
