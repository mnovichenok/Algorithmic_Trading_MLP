# Algorithmic Trading with MLP
This project implements an algorithmic trading system that predicts trading signals using a Multi-Layer Perceptron (MLP) written from scratch with Python and NumPy. It leverages historical stock data and technical indicators to generate predictions, with a containerized FastAPI backend and an interactive frontend built with Streamlit.

## Model Inputs
*The model is trained using key technical indicators derived from historical price data:*

#### - RSI (Relative Strength Index)
*Measures the momentum of recent price changes to evaluate overbought or oversold conditions.*

#### - MACD (Moving Average Convergence Divergence)
*Tracks the relationship between two moving averages to identify trend changes and momentum.*

#### - SMA (Simple Moving Averages)
*Smooths out price data to highlight trends; I have included 20-day and 50-day windows in this project.*

#### - Bollinger Bands
*Measures price volatility by placing bands above and below a moving average, indicating potential overbought/oversold zones.*

These indicators are computed using the Python ta library; their values form the feature set for the MLP

## Model Highlights
Custom MLP implementation with:

- Flexible architecture (configurable layer sizes)

- One-hot encoded multi-class output (Strong Sell → Strong Buy)

- Mini-batch gradient descent

- Softmax output layer

Signal labels are generated based on future returns over a user-defined time window

## Interface and Architecture
FastAPI backend processes predictions and model inference

Streamlit frontend allows users to input stock tickers and prediction horizons

Deployed via Podman containers, isolated by a custom network for internal service communication


## Installation & Running the Project
###  Prerequisites

- [Podman](https://podman.io/) installed and configured
- Podman machine (for macOS users):
```bash
podman machine init && podman machine start
```

#### 1. Clone the repository

```bash
git clone https://github.com/mnovichenok/Algorithmic_Trading_MLP.git
cd Algorithmic_Trading_MLP
```

#### 2. Create a shared network and build the containers

```bash
podman network create mlp-net
podman build -f Containerfile -t algo_trading_model .
podman build -f Containerfile.ui -t algo_trading_model_ui .
```

#### 3. Run the Containers

```bash
podman run -d --rm --name algo_logic --network mlp-net algo_trading_model
podman run -d --rm --name algo_ui --network mlp-net -p 8501:8501 algo_trading_model_ui
```

#### 4. Access the App

http://localhost:8501

### -Maya Novichenok

