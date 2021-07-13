import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


if __name__ == '__main__':
    
    movies = pd.read_csv('./in/clean_movies.csv')
    
    X = movies.drop(['movie_title', 'worldwide_gross'], axis=1)
    y = movies['worldwide_gross']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

    reg = GradientBoostingRegressor()
    results = cross_validate(
        reg, 
        X_train, 
        y_train, 
        cv=5, 
        scoring='r2', 
        return_train_score=True
    )    
    
    train_scores = results['train_score']
    test_scores = results['test_score']

    print('CVTrain: ', np.mean(train_scores))
    print('CVTest: ', np.mean(test_scores))

    reg.fit(X_train, y_train)
    print('Test: ', reg.score(X_test, y_test))
    