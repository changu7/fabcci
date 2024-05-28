import streamlit as st
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Streamlit app layout
st.title('Time Series Prediction App')

# Step 1: Upload CSV File
st.write("## Upload your CSV file")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the CSV file
    data = pd.read_csv(uploaded_file)
    st.write("## Uploaded Data")
    st.write(data)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

if uploaded_file is not None:
    # Assuming the CSV has columns: 'Date', 'External1', 'External2', 'External3', 'Target'
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Creating lag features for the past 12 months
    for i in range(1, 13):
        data[f'Target_lag_{i}'] = data['Target'].shift(i)
        data[f'External1_lag_{i}'] = data['External1'].shift(i)
        data[f'External2_lag_{i}'] = data['External2'].shift(i)
        data[f'External3_lag_{i}'] = data['External3'].shift(i)

    data.dropna(inplace=True)

    # Split data into features and target
    X = data.drop(columns=['Target'])
    y = data['Target']

    # Split data into training and test sets (using last 10% for testing)
    test_size = int(len(data) * 0.1)
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    # Step 2: Train SVR model with hyperparameter optimization using TimeSeriesSplit
    param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
    svr = SVR(kernel='rbf')
    tscv = TimeSeriesSplit(n_splits=10)
    
    grid_search = GridSearchCV(svr, param_grid, cv=tscv, scoring='neg_mean_absolute_error')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_

    # Display GridSearch results
    st.write("## GridSearchCV Results")
    st.write("Best parameters found: ", grid_search.best_params_)
    st.write("Grid scores on training set:")
    for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
        st.write(f"{params}: {mean_score:.3f}")

    # Evaluate the model using MAPE with TimeSeriesSplit
    tscv_splits = tscv.split(X_train)
    mape_scores = []
    for train_index, val_index in tscv_splits:
        X_train_split, X_val_split = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_split, y_val_split = y_train.iloc[train_index], y_train.iloc[val_index]
        best_model.fit(X_train_split, y_train_split)
        y_val_pred = best_model.predict(X_val_split)
        mape_scores.append(mean_absolute_percentage_error(y_val_split, y_val_pred))
    
    avg_mape = np.mean(mape_scores)
    st.write(f"Model MAPE: {avg_mape:.2f}%")
    
    # Step 3: Predict next 12 months using the last known 12 months data
    last_12_months = data.iloc[-12:].copy()
    future_predictions = []
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    best_model.fit(X_train_scaled, y_train)
    
    for i in range(12):
        future_features = last_12_months.drop(columns=['Target']).iloc[i:i+1]
        future_features_scaled = scaler.transform(future_features)
        prediction = best_model.predict(future_features_scaled)
        future_predictions.append(prediction[0])
        # Update last_12_months with the new prediction
        new_row = future_features.iloc[0].copy()
        new_row['Target'] = prediction[0]
        new_row = new_row.to_frame().T
        last_12_months = pd.concat([last_12_months, new_row])
        last_12_months = last_12_months.iloc[1:]

    future_dates = pd.date_range(start=data.index[-1], periods=13, freq='M')[1:]
    predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted_Target': future_predictions})
    predictions_df.set_index('Date', inplace=True)
    
    st.write("## Predictions for the next 12 months")
    st.write(predictions_df)
    
    # Plotting the predictions
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Target'], label='Actual')
    ax.plot(predictions_df.index, predictions_df['Predicted_Target'], label='Predicted', linestyle='--')
    ax.legend()
    st.pyplot(fig)
    
    # Step 4: Download predictions as CSV
    st.write("## Download predictions as CSV")
    csv = predictions_df.to_csv().encode('utf-8')
    st.download_button("Download CSV", csv, "predictions.csv", "text/csv")
    
    # Step 5: Return to initial page
    if st.button('Go back to upload page'):
        st.experimental_rerun()
