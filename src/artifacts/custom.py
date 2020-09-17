def transform(data, model):
    """
    Note: This hook may not have to be implemented for your model.
    In this case implemented for the model used in the example.
    Modify this method to add data transformation before scoring calls. For example, this can be
    used to implement one-hot encoding for models that don't include it on their own.
    Parameters
    ----------
    data: pd.DataFrame
    model: object, the deserialized model
    Returns
    -------
    pd.DataFrame
    """
    # Execute any steps you need to do before scoring

    pipeline = joblib.load('artifacts/preprocessing_pipeline_18-09-2020.pkl')
    data = pipeline.transform(data)
    return data


def load_model():
    model = keras.models.load_model('artifacts/classifier_18-09-2020.h5')

    return model