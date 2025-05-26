import streamlit as st
import requests

st.title("Algorithmic Trading Indicator")

ticker = st.text_input("Stock Ticker")
future_days = st.number_input("Number of Days into the Future", min_value=0)

if st.button("Determine Optimal Trading Strategy"):
    input = {
        "Ticker": ticker,
        "Future_Days": future_days
    }
    
    result = requests.post("http://algo_logic:8000/input/", json=input) #algo_trading_logic

    if result.status_code == 200:
        signal_map = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
        signal = signal_map[result.json()["signal"]]    
        st.success(f"Predicted signal: {signal}")
    
    
    else:
        #st.error(f"Error from API :  {result.status_code} — {result.text}")
        try:
            detail = result.json().get("detail", "Unknown error")
        
            if isinstance(detail, list):  # for validation errors
                detail = detail[0].get("msg", "Validation error")
        
        except Exception:
            # fallback if server didn't return JSON at all
            detail = result.text or "Server returned invalid response"
        
        st.error(f"API error ({result.status_code}): {detail}")



