import pathlib
import pytest

@pytest.fixture
def fixtures_dir():
    return pathlib.Path(__file__).parent / "fixtures"

@pytest.fixture
def mock_hdf5_path(fixtures_dir):
    return fixtures_dir / "mock_data.h5"
