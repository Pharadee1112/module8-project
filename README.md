This project is a web-based application for analyzing and forecasting PM2.5 air pollution levels. Users can select a city, explore historical air quality data, and visualize trends through interactive charts. The system also allows users to generate short-term PM2.5 predictions for the next 72 hours.

The backend is built using Flask, which handles data processing, model execution, and API routing. The system loads PM2.5, temperature, and wind data, merges them by datetime, and performs feature engineering (hour, month, day of week).

A Random Forest Regressor is used for prediction. Model performance is evaluated using Mean Absolute Error (MAE). The application demonstrates the integration of machine learning with a web-based interface for real-world air quality analysis.
