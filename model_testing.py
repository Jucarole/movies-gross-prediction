import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


if __name__ == '__main__':
    
    movies = pd.read_csv('./in/clean_movies.csv')
    
    X = movies.drop(['movie_title', 'worldwide_gross'], axis=1)
    y = movies['worldwide_gross']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

    params = {
        'kernel' : ['linear', 'rbf']
    }
    
    reg = GradientBoostingRegressor().fit(X_train,y_train)
    #gsearch = GridSearchCV(reg, param_grid=params, scoring='r2', cv=5)
    #gsearch.fit(X_train, y_train)

    #print(gsearch.best_score_)
    #print(gsearch.best_estimator_)
    
    
    y_pred = reg.predict(X_test)
    score_t = r2_score(y_test, y_pred)

    print(reg.score(X_test,y_test))
    print(score_t)
