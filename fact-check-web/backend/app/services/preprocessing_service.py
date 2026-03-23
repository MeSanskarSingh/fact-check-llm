from ml_pipeline.preprocessing.PreprocessingScript import EnhancedFactCheckPreprocessor

preprocessor = EnhancedFactCheckPreprocessor()


def preprocess_input(input_data):
    result = preprocessor.process_input(input_data)

    if result.status != "success":
        raise Exception(result.error_message)

    return result.text