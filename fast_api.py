from fastapi import FastAPI
from pydantic import BaseModel
import requests
from indicator import Indicator

app = FastAPI()
class Stock_Data(BaseModel) :
    Ticker: str
    Future_Days: int

@app.post("/input/")
def get_input(input: Stock_Data):
    model = Indicator(input.Ticker, input.Future_Days)
    signal = model.predict_signal()    
    return {"signal": signal}
    
