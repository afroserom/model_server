from typing import Optional
from fastapi import FastAPI
import os
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from enum import Enum
import pickle
import pandas as pd
import json

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
    CODE_GENDER: GenderEnum
    FLAG_OWN_CAR: YesNoEnum
    FLAG_OWN_REALTY: YesNoEnum
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: float
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    FLAG_MOBIL: FlagEnum
    FLAG_WORK_PHONE: FlagEnum
    FLAG_PHONE: FlagEnum
    FLAG_EMAIL: FlagEnum
    OCCUPATION_TYPE: str
    CNT_FAM_MEMBERS: int

API_VERSION = os.getenv("API_VERSION","v1")

app = FastAPI()

with open("utils/model.pkl", "rb") as file:
    model = pickle.load(file)

@app.get(f"/{API_VERSION}/predict")
async def predict(observation:Observation, proba:bool = True):
    observation_json = jsonable_encoder(observation)
    print(observation_json)
    df = pd.DataFrame([observation_json])
    print(df)
    print(model.predict_proba(df))

    return {
        "prediction": model.predict_proba(df)[:,1][0] if proba else model.predict(df)[0],
        "features":  observation
    }