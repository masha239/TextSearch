import pytest
import numpy as np
import requests.exceptions

from src.extract import process_url


def test_constuct_filenode_dir():
    url = "http://localhost"
    res = process_url(url)
    assert res[0] == 200
    assert type(res[1]) == np.ndarray
    assert res[1].shape == (1, 13)


def test_wrong_type():
    url = 1
    pytest.raises(requests.exceptions.MissingSchema, process_url, url)


def test_unable_url():
    url = 'http://yandex.ru/facebook'
    res = process_url(url)
    assert res[0] == 403
