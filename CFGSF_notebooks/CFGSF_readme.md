The Corporación Favorita Sales Forecasting Project aims to create a sales forecasting model for one of Ecuador's largest grocery retailers. 

The main goal is to predict daily sales for many items across various stores to improve inventory management and business planning. 

The project began with Exploratory Data Analysis (EDA) to analyze the sales data, revealing significant volatility and strong seasonality in sales, with clear weekly and monthly cycles. The use of rolling statistics helped in understanding trends and variability over time. 

Next, feature engineering and modeling was carried out. Features were created from the raw data, including time-based and lag features, and incorporated external factors like promotions and holidays. 

Although a SARIMA model was initially used, a more advanced XGBoost model was ultimately selected for its ability to handle complex features and deliver robust sales predictions. 

The project culminated in the development of a Streamlit application, which allows users to forecast sales for specific items and stores, view interactive charts, and download forecast data.  
