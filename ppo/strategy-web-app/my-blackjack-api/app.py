from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from blackjack_env import BlackjackEnvCountingFirstMove
from stable_baselines3 import PPO
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Load model and env once at startup
env = BlackjackEnvCountingFirstMove()
model = PPO.load("models/final_model.zip", env=env)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Input schema
class StateInput(BaseModel):
    observation: list  # list of 13 integers

@app.get("/")
def root():
    return {"message": "Blackjack AI is live!"}

@app.post("/predict")
def predict(state: StateInput):
    try:
        obs = np.array(state.observation)
        action, _ = model.predict(obs, deterministic=True)
        return {"action": int(action)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
