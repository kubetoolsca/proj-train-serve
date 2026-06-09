"""Smoke tests — verify the package is importable."""


def test_smoke():
    assert True


def test_import_model():
    from src.image_classifier import SimpleCNN

    assert SimpleCNN is not None


def test_import_data():
    from src.image_classifier import create_dataloaders

    assert create_dataloaders is not None


def test_import_evaluate():
    from src.image_classifier import evaluate

    assert evaluate is not None


def test_import_labels():
    from src.image_classifier import FASHION_MNIST_LABELS

    assert len(FASHION_MNIST_LABELS) == 10


def test_import_metrics():
    from src.image_classifier import EvalMetrics

    assert EvalMetrics is not None
