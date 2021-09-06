from typing import Optional
from fastapi import FastAPI
import os
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from enum import Enum
import pickle
import pandas as pd
import json
from typing import List, Optional

class GenderEnum(str, Enum):
    male = "M"
    female = "F"

class YesNoEnum(str, Enum):
    yes = "Y"
    no = "N"

class FlagEnum(int, Enum):
    yes = 1
    no = 0

class Observation(BaseModel):
    CODE_GENDER: GenderEnum = None
    FLAG_OWN_CAR: YesNoEnum = None
    FLAG_OWN_REALTY: YesNoEnum = None
    CNT_CHILDREN: int = None
    AMT_INCOME_TOTAL: float = None
    NAME_INCOME_TYPE: str = None
    NAME_EDUCATION_TYPE: str = None
    NAME_FAMILY_STATUS: str = None
    NAME_HOUSING_TYPE: str = None
    DAYS_BIRTH: int = None
    DAYS_EMPLOYED: int = None
    FLAG_MOBIL: FlagEnum = None
    FLAG_WORK_PHONE: FlagEnum = None
    FLAG_PHONE: FlagEnum = None
    FLAG_EMAIL: FlagEnum = None
    OCCUPATION_TYPE: str = None
    CNT_FAM_MEMBERS: int = None

class Observations(BaseModel):
    instances: List[Observation]

class PredictionOut(BaseModel):
    prediction: float
    features: Observation

class PredictionsOut(BaseModel):
    prediction: List[float]

API_VERSION = os.getenv("API_VERSION","v1")

app = FastAPI()

with open(os.path.join(os.path.dirname(__file__), "utils/model.pkl"), "rb") as file:
    model = pickle.load(file)

@app.post(f"/{API_VERSION}/predict", response_model=PredictionOut)
async def predict(observation:Observation, proba:bool = True):
    observation_json = jsonable_encoder(observation)
    df = pd.DataFrame([observation_json])
    return {
        "prediction": model.predict_proba(df)[:,1][0] if proba else model.predict(df)[0],
        "features":  observation
    }

@app.post(f"/{API_VERSION}/batch_predict", response_model=PredictionsOut)
async def batch_predict(observations:Observations, proba:bool = True):
    observations_json = jsonable_encoder(observations.instances)
    df = pd.DataFrame(observations_json)
    return {
        "prediction": list(model.predict_proba(df)[:,1]) if proba else list(model.predict(df)),
    }