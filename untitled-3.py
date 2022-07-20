import uvicorn
from fastapi import Request, FastAPI
from typing import Optional
from pydantic import BaseModel
import pickle as pkl
import numpy as np
import pandas as pd
import json
print("Step 1: libraries dowloaded successfully")

# Loading predict models and last database
predict_model_1 = pkl.load(open("saved_model.pkl","rb"))
predict_model_2 = pkl.load(open("prob_model_2.pkl","rb"))
df_last_database = pd.read_csv('base_2656.csv', sep=',')
print("Step 2: models and database owloaded successfully")

# esteblishing values of question numbers for every exam
dict_exam_dates = {
    24: 55,
    70: 25,
    86: 25,
    90: 29,
    92: 17,
    93: 14,
    94: 17,
    96: 14,
    97: 25,
    98: 17,
    99: 26,
    100: 48,
    101: 60,
    102: 50,
    103: 16,
    104: 17,
    105: 180,
    2102: 84,
    2105: 16,
    2106: 100,
    3840: 9,
    3900: 20,
    3952: 11,
    3953: 11,
    3954: 11,
    3955: 11,
    3956: 11,
    3957: 14,
    3958: 9,
    3959: 11,
    3960: 11,
    3961: 9,
    3962: 6,
    3963: 6,
    4183: 9,
    4184: 12,
    4185: 6,
    4186: 12,
    4187: 12,
    4188: 18,
    4189: 9,
    4190: 9,
    4191: 15,
    4192: 12,
    4193: 15,
    4194: 9,
    4195: 12,
    4196: 6,
    4197: 9,
    4198: 9,
    4199: 9,
    4218: 11,
    4219: 8,
    4220: 6,
    4221: 5
}
print("Step 3: dict of exams dowloaded successfully")

with open('input_2656.json', 'r') as fale:
    data = json.load(fale)
    print(data)

print("Step 7: Request preformated (data.json()). Type =", type(data))
print("input data")
a =  input()
print(type(a))



c = json.loads(json.dumps(a))
print(type(c))
#print("Step 8: Request formated (json.loads(a)). Type =", type(c))
