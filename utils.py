import pandas as pd
import joblib


class Utils:

    def load_from_csv(self, path):
        return pd.read_csv(path)

    def features_target(self, dataset, drop_cols, target):
        X = dataset.drop(drop_cols, axis=1)
        y = dataset[target]
        return X, y

    def model_export(self,file_name, model):
        path = f'./models/{file_name}.pkl'
        joblib.dump(model, path)

    def add_model_info(self,model_file_name, params, score):
        f = open('./models/models_info.txt', 'a')
        f.write(f'{model_file_name}: \n')
        f.write(f'Parameters:  {params} \n')
        f.write(f'Score: {score} \n')
        f.close()
