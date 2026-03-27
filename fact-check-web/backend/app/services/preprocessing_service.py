from ml_pipeline.preprocessing.PreprocessingScript import EnhancedFactCheckPreprocessor

preprocessor = None

def get_preprocessor():
    global preprocessor
    if preprocessor is None:
        from ml_pipeline.preprocessing.PreprocessingScript import EnhancedFactCheckPreprocessor
        preprocessor = EnhancedFactCheckPreprocessor()
    return preprocessor


def preprocess_input(input_data):
    preprocessor = get_preprocessor()
    result = preprocessor.process_input(input_data)

    if result.status != "success":
        raise Exception(result.error_message)

    return result.text


def preprocess_input(input_data):
    result = preprocessor.process_input(input_data)

    if result.status != "success":
        raise Exception(result.error_message)

    return result.text