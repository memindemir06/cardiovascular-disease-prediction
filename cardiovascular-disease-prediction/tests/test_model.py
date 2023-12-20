import pytest
from src.model import CardiovascularDiseaseModel

def test_train():
    model = CardiovascularDiseaseModel()
    # Assuming you have a function to get training data
    X_train, y_train = get_training_data() 
    model.train(X_train, y_train)
    assert model.is_trained == True

def test_predict():
    model = CardiovascularDiseaseModel()
    # Assuming you have a function to get test data
    X_test = get_test_data() 
    predictions = model.predict(X_test)
    assert predictions is not None

def test_evaluate():
    model = CardiovascularDiseaseModel()
    # Assuming you have a function to get test data
    X_test, y_test = get_test_data() 
    model.train(X_test, y_test)
    score = model.evaluate(X_test, y_test)
    assert 0 <= score <= 1