g_code_dir = None
def init(code_dir):
    global g_code_dir
    g_code_dir = code_dir

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

    pipeline_path = 'preprocessing_pipeline_18-09-2020.pkl'
    pipeline = joblib.load(os.path.join(g_code_dir, pipeline_path))
    data = pipeline.transform(data)
    return data

def load_model(code_dir):
    model_path = 'classifier_18-09-2020.h5'
    model = keras.models.load_model(os.path.join(code_dir, model_path))
    return model