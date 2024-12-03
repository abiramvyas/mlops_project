import xgboost as xgb
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_absolute_error, r2_score,mean_squared_error
import pandas as pd

def train_model(X, y, algorithm):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if algorithm == "XGBoost":
        model = xgb.XGBRegressor()
    elif algorithm == "Linear Regression":
        model = LinearRegression()
    elif algorithm == "Lasso":
        model = Lasso()
    elif algorithm == "Random Forest":
        model = RandomForestRegressor(max_depth=6)
    elif algorithm == "Extra Trees Regressor":
        model = ExtraTreesRegressor(max_depth=6)
    else:
        raise ValueError("Unsupported algorithm")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Calculate feature importance or coefficients
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importance = model.coef_

    # Record feature importance or coefficients
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    # Prepare response data
    response_data = {
        "MSE": mean_squared_error(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "R2": r2_score(y_test, predictions),
        "predictions": predictions.tolist(),
        "feature_importance": feature_importance_df.to_dict(orient='records')
    }

    return response_data
