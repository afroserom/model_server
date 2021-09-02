ETL_VERSION="0.0.1"
MODEL_VERSION="0.0.1"
DATASET_TRAIN_FILENAME="dataset_train.parquet"
DATASET_VALIDATION_FILENAME="dataset_validation.parquet"
PREPROCESSOR_FILENAME=f"preprocessor_{ETL_VERSION}.pkl"
BEST_ESTIMATOR_FILENAME=f"best_estimator_{ETL_VERSION}.pkl"
TRAINED_BEST_ESTIMATOR_FILENAME=f"trained_best_estimator_{ETL_VERSION}.pkl"
MODEL_FILENAME=f"model.pkl"

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
def print_settings(name):
    for variable in dir(name):
        if not variable.startswith('__'):
            print(color.BOLD+color.RED+variable, ':', getattr(name, variable))
    print(color.END)