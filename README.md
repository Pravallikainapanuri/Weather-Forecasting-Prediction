# Weather-Forecasting-Prediction
# 🌦️ Weather Forecasting Using CNN and XGBoost

This project leverages **Convolutional Neural Networks (CNN)** and **Extreme Gradient Boosting (XGBoost)** to predict weather suitability for outdoor activities such as BBQs. The system performs binary classification based on historical weather data.

---

## 📁 Project Structure


---

## 📊 Dataset Description

- **weather_prediction_dataset2.csv**:  
  Contains daily meteorological parameters like temperature, humidity, wind speed, visibility, etc.

- **weather_prediction_bbq_labels.csv**:  
  Includes the binary target column: `BASEL_BBQ_weather`  
  - `1`: Suitable for BBQ  
  - `0`: Not suitable

The two datasets are merged using the `DATE` column.

---

## 🧠 Models Used

### ✅ CNN Model
- Used for feature extraction and temporal pattern recognition.
- Input reshaped to 3D (`samples, features, 1`) for Conv1D.
- Architecture: Conv1D → Pooling → Dropout → Dense Layers

### ✅ XGBoost Model
- Structured data model using gradient-boosted decision trees.
- GridSearchCV used for hyperparameter tuning.
- Interpretable through feature importance.

---

## 🛠️ Requirements

Install all required Python packages using:

```bash
pip install -r requirements.txt
git clone https://github.com/your-username/weather-forecasting-cnn-xgboost.git
cd weather-forecasting-cnn-xgboost
python cnn_weather_forecast.py
python weather_xgboost_tuned_model.py
