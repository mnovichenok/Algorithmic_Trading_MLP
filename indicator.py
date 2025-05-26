import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from MLP import MLP

class Indicator:

    def __init__(self, ticker, future_days):
        self.ticker = ticker
        self.future_days = future_days
        self.model = None
        self.scaler = None
        self.features = [
            "rsi", "macd", "macd_signal",
            "sma_20", "sma_50", "bb_high", "bb_low"
        ]
        self.train_model()

    def label_signal(self,r):
        if r < -0.02:
            return 0  # Strong Sell
        elif r < -0.005:
            return 1  # Sell
        elif r <= 0.005:
            return 2  # Hold
        elif r <= 0.02:
            return 3  # Buy
        else:
            return 4  # Strong Buy      
        #Thresholds based on : Machine Learning for Algorithmic Trading – Stefan Jansen (O’Reilly, 2020)
        

    def feature_engineering(self, df):
        close = df["Close"].squeeze()  # guarantees Series (1D)
        df["rsi"] = RSIIndicator(close=close).rsi() #measures momentum; Overbought/Oversold conditions
        
        macd = MACD(close=close) #shows trend strength and direction
        macd_line = macd.macd()
        macd_signal = macd.macd_signal()
        df["macd"] = pd.Series(np.ravel(macd_line), index=df.index)
        df["macd_signal"] = pd.Series(np.ravel(macd_signal), index=df.index)
        
        # SMA (manual with pandas)  
        df["sma_20"] = df["Close"].rolling(window=20).mean() #short-term trend indicator
        df["sma_50"] = df["Close"].rolling(window=50).mean() #medium-term trend indicator
        
        # Bollinger Bands (manual) *measure volatility and extremes
        rolling_mean = df["Close"].rolling(window=20).mean()
        rolling_std = df["Close"].rolling(window=20).std()
        df["bb_high"] = rolling_mean + 2 * rolling_std
        df["bb_low"] = rolling_mean - 2 * rolling_std
        
        df = df.dropna()
        return df
    
    def train_model(self):
        df = yf.download(self.ticker, start="2020-01-01", end="2024-12-31")
        #TODO set END to current date
        df = self.feature_engineering(df)
        
        # If MultiIndex, flatten it
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        #df["future_return"] = df["Close"].shift(-self.future_days) / df["Close"] - 1
        #future_return_t​ = ( Close_(t+Δt) / Close_t​ ​​) − 1

        df["future_return"] = df["Close"].shift(-self.future_days).rolling(window=10).mean() / df["Close"] - 1
        
        
        df["Signal"] = df["future_return"].apply(self.label_signal)
                
        df.dropna(inplace=True)
    
    
        X = df[self.features].values
        y = df['Signal'].astype(int).values

    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101, stratify=y)
    
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
       
        #Predict
        num_classes = int(np.max(y_train)) + 1
    
        #TODO Grid seach for optimal parameter combination
        
        input_size = len(self.features)
        output_size = num_classes  #5 (Strong Sell → Strong Buy)
        self.model = MLP(
            nbr_layers=4,
            units_per_layer=(input_size, 64, 32, output_size) 
        )

        y_train = self.model.one_hot_encoded(y_train, num_classes) #One-hot encode y
        
        self.model.fit(X_train, y_train, 0.01, 500000, 2, 100000) #lr, epochs, minibatch_size, print_updates

        
        self.X_test = X_test
        self.y_test = y_test

    def predict_signal(self):
        df = yf.download(self.ticker, period="3mo")
        df = self.feature_engineering(df).dropna()
        latest = df[self.features].iloc[-1]
        x = self.scaler.transform([latest])
        return int(np.argmax(self.model.predict(x[0])))
        