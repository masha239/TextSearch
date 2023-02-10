from src.app import transform_answer


def test_transform_answer():
    assert transform_answer(1) == 'Yes'
    assert transform_answer(0) == 'No'
