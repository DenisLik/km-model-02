# ======== model_2.3 ==== 07/19/2022 ====

# Importing Necessary modules
import uvicorn
from fastapi import Request, FastAPI
from typing import Optional
from pydantic import BaseModel
import pickle as pkl
import numpy as np
import pandas as pd
import json

# Loading predict models and last database
predict_model_1 = pkl.load(open("saved_model.pkl","rb"))
predict_model_2 = pkl.load(open("prob_model_2.pkl","rb"))
df_last_database = pd.read_csv('base_2656.csv', sep=',')

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


# Declaring our FastAPI instance for further application start
app_for_models = FastAPI()

# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    user_id : int
    temp_id: int
    exam_id : int
    try_id : int
    items : Optional[list] = None   


# Creating an Endpoint to receive the data to make prediction on
@app_for_models.post('/predict')
def predict(data: request_body):
    
    
    # STEP 1. Gettig df from input request
        
    def func_condition(any_dict):
        if any_dict['user_id'] != 0 and any_dict['exam_id'] in list(dict_exam_dates.keys()):
            print("Condition is True")
            return True
        elif any_dict['user_id'] == 0 and any_dict['exam_id'] == 24:
            print("Condition is True")
            return True
        else:
            print("Condition is False")
            return False
    
    a = data
    b = data.json()
    c = json.loads(b)
       
    
    # ================= step 1 ====================
    
    
    
    
    if func_condition(c):
        
        
        def get_df_from_input(input_dict):
            dict_input = dict(user_id=input_dict["user_id"],
                              temp_id=input_dict["temp_id"],
                              exam_id=input_dict["exam_id"],
                              try_id=input_dict["try_id"])
            df_input_left = pd.DataFrame(data=dict_input,
                                 columns=["user_id", "temp_id", "exam_id", "try_id"],
                                 index=range(len(input_dict["items"])))
            df_input_right = pd.DataFrame(data=input_dict["items"],
                                  columns=["question_id", "answer_date", "user_answer", "right_answer"])
            df_input = pd.concat([df_input_left, df_input_right], axis=1)

    
            def func_transform(x):
                y = ''.join(str(i) for i in sorted(x.split(',')))
                return y
    
            df_input["user_answer"] = df_input["user_answer"].transform(func_transform)
            df_input["right_answer"] = df_input["right_answer"].transform(func_transform)
            df_input["result"] = 0
            df_input["result"] = np.where(df_input["user_answer"] == df_input["right_answer"], 1, 0)
    
            user_id = dict_input["user_id"]
            exam_id = dict_input["exam_id"]
    
            def func_result(x):
                a = x.sum()
                b = dict_exam_dates[exam_id]
                result = round(a / b, 4)
                return result
    
            def func_counter(y):
                return dict_exam_dates[exam_id]

            df_input_grouped = df_input.copy().groupby(["user_id", "temp_id", "exam_id", "try_id"], as_index=False).agg({"question_id": func_counter, "result": func_result})
            result = df_input_grouped["result"][0]
    
    
            get_answer = (user_id, result, df_input, df_input_grouped)
            return get_answer         
        
        
        get_answer = get_df_from_input(c)
    

        # STEP 2. Loading the data of user from last database
        def choosing_user(user, df):
            user_id = "user_id == " + str(user)
            df_base_user = df.query(user_id).copy()
            
            def func_transform(x):
                y = ''.join(str(i) for i in sorted(x.split(',')))
                return y
            
            df_base_user["user_answer"] = df_base_user["user_answer"].transform(func_transform)
            df_base_user["right_answer"] = df_base_user["right_answer"].transform(func_transform)
            df_base_user["result"] = np.where(df_base_user["user_answer"] == df_base_user["right_answer"], 1, 0)

            return df_base_user    

        df_base_user = choosing_user(get_answer[0], df_last_database)

    
        # STEP 3. Grouping all df's to the total database of the user
        def func_mean_result(x):
            result_indexes = x.index.tolist()
            exam_id_indx = df_base_user.loc[result_indexes, "exam_id"].head(1)
            exam_id = exam_id_indx.iloc[0]
            a = x.sum()
            b = dict_exam_dates[exam_id]
            mean_result = round(a / b, 4)
            return mean_result
        
        def func_counter(x):
            result_indexes = x.index.tolist()
            exam_id_indx = df_base_user.loc[result_indexes, "exam_id"].head(1)
            exam_id = exam_id_indx.iloc[0]
            return dict_exam_dates[exam_id]
    
        df_base_user_grouped = df_base_user \
            .copy() \
            .groupby(["user_id", "temp_id", "exam_id", "try_id"], as_index=False) \
            .agg({"question_id": func_counter, "result": func_mean_result})    
    
        def df_concating(x, y):
            df_total = pd.concat([x, y],axis = 0)
            return df_total
    
        df_total_user = df_concating(df_base_user_grouped, get_answer[3])

    
        # STEP 4. Calculating values for the user for further ML
        amount_of_questions_passed = df_total_user["question_id"].sum().item()
        overal_mean_result = round(df_total_user["result"].mean(), 4).item()
        amount_of_exams = df_total_user["try_id"].count().item()
    
        input_values = {
            "base_1": amount_of_questions_passed,
            "result_1": overal_mean_result,
            "try_id_1": amount_of_exams
        }

        # input data for model 1
        df_input_values = pd.DataFrame(input_values, index = range(1))

        # input data for model 2
        input_value = np.array(df_input_values["result_1"]).reshape(-1, 1)

    
        # STEP 5. Predictions via Machine Learning
        prediction_1 = predict_model_1.predict(df_input_values)
        prediction_2 = predict_model_2.predict(input_value)
        output_1 = int(prediction_1[0])
        output_2 = round(float(prediction_2[0]), 2)

    
        # STEP 6. Creating answer for KnowLedgemap
        dict_output = {"amount_of_questions_passed": amount_of_questions_passed,
                       "amount_of_questions_lost": output_1,
                       "overal_mean_result": overal_mean_result * 100,
                       "probability": output_2}
       
    
        # STEP 7. Creating new version of the last database
        df_total_base = df_concating(df_last_database, get_answer[2])
        df_total_base.to_csv('base_2656.csv', index=False)

        return dict_output
    
    
    else:

        return
    
        



#uvicorn.run(app_for_models)
