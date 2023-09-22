import joblib

def predict(data, model):

    model = joblib.load('random_fores_model.pkl')

    pipeline = joblib.load("full_pipeline.pkl")
    data = pipeline.transform(data)
    
    return model.predict(data)