from utils import Utils
from models import Models


if __name__ == '__main__':
    utils = Utils()
    models = Models()

    data = utils.load_from_csv('./in/clean_movies.csv')
    drop_cols = ['worldwide_gross', 'movie_title']
    X, y = utils.features_target(data, drop_cols, 'worldwide_gross')

    score, params, model = models.grid_training(X, y)

    # utils.model_export('model_1',model)
    utils.add_model_info('model_1',params,score)