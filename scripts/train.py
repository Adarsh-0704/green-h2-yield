import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score
from pipeline import data_engineering

def train_and_test():
    # Making sure to get the trained spline
    df, train_spline = data_engineering('../data/training_data.csv')
    features = ['GHI(W/m2)', 'Windspeed(m/s)', 'Stored Energy(MWh)', 'Mon',
                'Day', 'spline_hr_1','spline_hr_2', 'spline_hr_3',
                'spline_hr_4', 'Windspeed_mean_3h','Windspeed_std_3h',
                'GHI_mean_3h', 'Windspeed_lag_1hr','GHI_lag_1hr'
                ]
    X = df[features]
    y = df['Hydrogen_yield(kg)']
    # Hyperparameter grid for optimization made sure to keep max_depth low
    # and increasing min_samples_leaf to prevent overfitting
    parameters = {
        'n_estimators' : [150, 200, 250, 300, 350],
        'max_depth' : [8, 10, 12, 15, 20],
        'min_samples_split' : [5, 10, 15, 20, 30],
        'min_samples_leaf' : [2, 4, 6, 8, 12, 16],
        'max_features' : ['sqrt', 'log2', 0.5]
    }

    rf = RandomForestRegressor(random_state=42)

    rf_rand = RandomizedSearchCV(
        estimator=rf, param_distributions=parameters,
        n_iter=10, cv=5, scoring='r2',
        verbose=2, random_state=42, n_jobs=-1
        )

    rf_rand.fit(X, y)
    # Getting the best model from the estimator
    best_rf_model = rf_rand.best_estimator_
    y_pred = best_rf_model.predict(X)

    rmse = root_mean_squared_error(y, y_pred)
    train_r2 = r2_score(y, y_pred)
    n = X.shape[0]
    p = X.shape[1]
    adjusted_r2 = 1 - ((1 - train_r2) * (n - 1) / (n - p - 1))
    # To check if rf ever overfitted
    print('SEARCH COMPLETE')
    print(f'Best Paramters are: {rf_rand.best_params_}')
    print(f'Best RMSE is: {rmse:.3f}kg')
    print(f'Best R2 Score is: {rf_rand.best_score_:.4f}')
    print(f'Train R2: {train_r2:.4f}')
    print(f'Adjusted R2: {adjusted_r2:.4f}')

    # Saving the models so the training and testing notebooks can replicate the process
    joblib.dump(best_rf_model, '../models/random_forest_regressor.joblib')
    joblib.dump(train_spline, '../models/training_spline_transformer.joblib')

    return best_rf_model, train_spline

train_rf, rf_spline = train_and_test()