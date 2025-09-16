from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

categorical_features = ['Make', 'Colour']
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_features = ['Odometer (KM)']
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

door_features = ['Doors']
door_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=4)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('cat', categorical_transformer, categorical_features),
    ('num', numeric_transformer, numeric_features),
    ('door', door_transformer, door_features)
])

models = {
    "Dummy": DummyRegressor(),
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

param_grids = {
    "Dummy": {}, 
    "LinearRegression": {}, 
    "RandomForest": {
        "model__n_estimators": [100, 200, 500],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4]
    },
    "GradientBoosting": {
        "model__n_estimators": [100, 200, 500],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [3, 5, 7],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4]
    }
}


x = car_df.drop('Price', axis=1)
y = car_df['Price']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

results_list= []

for name, model in models.items():
    print(f" Model: {name}")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    if param_grids[name]: 
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grids[name],
            n_iter=10,
            scoring='r2',
            n_jobs=-1,
            cv=5,
            verbose=1,
            random_state=42
        )
        search.fit(xtrain, ytrain)
        best_model = search.best_estimator_
        print("Best Params:", search.best_params_)
        print("Best CV R2:", search.best_score_)
    else:  
        pipeline.fit(xtrain, ytrain)
        best_model = pipeline

    y_pred = best_model.predict(xtest)


    results_list.append({
    "Model": name,
    "MAE": mean_absolute_error(ytest, y_pred),
    "MSE": mean_squared_error(ytest, y_pred),
    "R2": r2_score(ytest, y_pred)})

df_results = pd.DataFrame(results_list)
print(df_results)