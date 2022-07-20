# ======== model_2 ==== 07/14/2022 ====
# Importing Necessary modules
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


def get_df_from_input(input_dict):
    print("Step 10: type(input_dict) =", type(input_dict))
    dict_input = dict(user_id=input_dict["user_id"],
                      temp_id=input_dict["temp_id"],
                      exam_id=input_dict["exam_id"],
                      try_id=input_dict["try_id"])
    print("Step 11: dict_input created, type =", type(dict_input))
    
    df_input_left = pd.DataFrame(data=dict_input,
                                 columns=["user_id", "temp_id", "exam_id", "try_id"],
                                 index=range(len(input_dict["items"])))
    print("Step 12: df_input_left created, type =", type(df_input_left))
    
    df_input_right = pd.DataFrame(data=input_dict["items"],
                                  columns=["question_id", "answer_date", "user_answer", "right_answer"])
    print("Step 13: df_input_right created, type =", type(df_input_right))
    
    df_input = pd.concat([df_input_left, df_input_right], axis=1)
    print("Step 14: df_input concated, type =", type(df_input))
    
    def func_transform(x):
        # print(x)
        y = ''.join(str(i) for i in sorted(x.split(',')))
        return y
    
    df_input["user_answer"] = df_input["user_answer"].transform(func_transform)
    df_input["right_answer"] = df_input["right_answer"].transform(func_transform)
    df_input["result"] = 0
    df_input["result"] = np.where(df_input["user_answer"] == df_input["right_answer"], 1, 0)
    print("Step 15: df_input updated, type =", type(df_input), df_input.shape)
    
    user_id = dict_input["user_id"]
    print("Step 16: type of user_id =", type(user_id))
    exam_id = dict_input["exam_id"]
    print("Step 17: type of exam_id =", type(exam_id))
    
    def func_result(x):
        a = x.sum()
        b = dict_exam_dates[exam_id]
        result = round(a / b, 4)
        return result
    
    def func_counter(y):
        return dict_exam_dates[exam_id]

    df_input_grouped = df_input.copy().groupby(["user_id", "temp_id", "exam_id", "try_id"], as_index=False).agg({"question_id": func_counter, "result": func_result})
    print("Step 18: type of df_input_grouped =", type(df_input_grouped), df_input_grouped.shape)
    result = df_input_grouped["result"][0]
    
    
    get_answer = (user_id, result, df_input, df_input_grouped)
    print("Step 19: get_answer created, type =", type(get_answer))
    return get_answer    

#data = input()
#a = data.json()
a = input()

print("Step 7: Request preformated (input). Type =", type(a))
print(a)
c = json.loads(a)
print("Step 8: Request formated (json.loads(a)). Type =", type(c))
print("Step 9: Function get_df_from_input(c) started with Formated Request")
get_answer = get_df_from_input(c)
print('STEP 1 has finised')
print()



