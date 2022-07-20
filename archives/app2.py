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

# Declaring our FastAPI instance for further application start
app_for_models = FastAPI()
print("Step 4: app_for_models dowloaded successfully")

# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    user_id : int
    temp_id: int
    exam_id : int
    try_id : int
    items : Optional[list] = None   

print("Step 5: class request_body created successfully", type(request_body))

# Creating an Endpoint to receive the data to make prediction on.
@app_for_models.post('/predict')
def predict(data: request_body):
    print("Step 6: Request get successfully. Type =", type(data))
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
    print("Step 7: Request preformated (a = data). Type =", type(a))
    b = data.json()
    print("Step 8: Request formated (b = data.json()). Type =", type(b))
    c = json.loads(b)
    print("Step 8.1: Request formated (c = json.loads(b)). Type =", type(c))
    print("Step 8.2: c =", c)
    print("Step 8.3: dict_exam_dates.keys() =", list(dict_exam_dates.keys()))
    print("Step 8.4: c['user_id'] =", c['user_id'])
    print("Step 8.5: c['temp_id'] =", c['temp_id'])
    print("Step 8.6: c['exam_id'] =", c['exam_id'])
    print("Step 8.7: c['try_id'] =", c['try_id'])
    print("Step 9: Function get_df_from_input(c) started with Formated Request")
    
    
    # ================= step 1 ====================
    
    
    
    
    if func_condition(c):
        print("CONDITION #1. LEVEL #1")
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
        
        
        
        
        
        
        get_answer = get_df_from_input(c)
    
        #get_answer = get_df_from_input(c)
        print('STEP 1 has finised')
        print()
    
        # STEP 2. Loading the data of user from last database
        def choosing_user(user, df):
            print("Step 22: user and df get and types =", type(user), type(df))
            user_id = "user_id == " + str(user)
            df_base_user = df.query(user_id).copy()
            
            def func_transform(x):
                y = ''.join(str(i) for i in sorted(x.split(',')))
                return y
            
            df_base_user["user_answer"] = df_base_user["user_answer"].transform(func_transform)
            df_base_user["right_answer"] = df_base_user["right_answer"].transform(func_transform)
            df_base_user["result"] = np.where(df_base_user["user_answer"] == df_base_user["right_answer"], 1, 0)
            print("Step 23: df_base_user transformed, type =", type(df_base_user), df_base_user.shape)
            
            
            return df_base_user    
        print("Step 21: function of choosing_user run")
        df_base_user = choosing_user(get_answer[0], df_last_database)
        print('STEP 2 has finised')
        print()    
    
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
        print("Step 31: df_base_user_grouped crated, type =", type(df_base_user_grouped), df_base_user_grouped.shape)
    
        def df_concating(x, y):
            df_total = pd.concat([x, y],axis = 0)
            return df_total
    
        df_total_user = df_concating(df_base_user_grouped, get_answer[3])
        print("Step 32: df_total_user crated, type =", type(df_total_user), df_total_user.shape)
        print('STEP 3 has finised')
        print()    
    
        # STEP 4. Calculating values for the user for further ML
        amount_of_questions_passed = df_total_user["question_id"].sum().item()
        overal_mean_result = round(df_total_user["result"].mean(), 4).item()
        amount_of_exams = df_total_user["try_id"].count().item()
        print("Step 41: values for input_values created")
    
        input_values = {
            "base_1": amount_of_questions_passed,
            "result_1": overal_mean_result,
            "try_id_1": amount_of_exams
        }
        print("Step 42: input_values created, type =", type(input_values))
        print("Step 42.1: amount_of_questions_passed, type =", type(amount_of_questions_passed))
        print("Step 42.2: overal_mean_result, type =", type(overal_mean_result))
        # input data for model 1
        df_input_values = pd.DataFrame(input_values, index = range(1))
        print("Step 43: df_input_values created from input_values, type =", type(df_input_values))
        # input data for model 2
        input_value = np.array(df_input_values["result_1"]).reshape(-1, 1)
        print("Step 44: input_value created from df_input_values, type =", type(input_value))
        print('STEP 4 has finised')
        print()    
    
        # STEP 5. Predictions via Machine Learning
        prediction_1 = predict_model_1.predict(df_input_values)
        print("Step 51: prediction_1 generated from predict_model_1, type =", type(prediction_1))
        prediction_2 = predict_model_2.predict(input_value)
        print("Step 52: prediction_2 generated from predict_model_2, type =", type(prediction_2))
        output_1 = int(prediction_1[0])
        print("Step 53: output_1 generated from prediction_1, type =", type(output_1))
        output_2 = round(float(prediction_2[0]), 2)
        print("Step 54: output_2 generated from prediction_2, type =", type(output_2))
        print('STEP 5 has finised')
        print()    
    
        # STEP 6. Creating answer for KnowLedgemap
        dict_output = {"amount_of_questions_passed": amount_of_questions_passed,
                       "amount_of_questions_lost": output_1,
                       "overal_mean_result": overal_mean_result * 100,
                       "probability": output_2}
        #dict_output = {}
        #dict_output["amount_of_questions_passed"] = amount_of_questions_passed
        #dict_output["amount_of_questions_lost"] = output_1
        #dict_output["overal_mean_result"] = overal_mean_result * 100
        #dict_output["probability"] = output_2   
      
        print("Step 61: dict_output created, type =", type(dict_output))
        print("Step 62: dict_output =", dict_output)
    
    
        # STEP 7. Creating new version of the last database
        df_total_base = df_concating(df_last_database, get_answer[2])
        df_total_base.to_csv('base_2656.csv', index=False)
        # json_output = json.dumps(dict_output)
        # print("Step 62: json_output created, type =", type(json_output))
    
        #dict_output_2 = {"amount_of_questions_passed": 32,
                       #"amount_of_questions_lost": output_1,
                       #"overal_mean_result": 2 * 100,
                       #"probability": output_2}     
        #print(dict_output)
        #print()
        #return dict_output
        return dict_output
    
    
    else:
        print("CONDITION #2. LEVEL #1")
        #dict_output = {}
        return
    
        



uvicorn.run(app_for_models)
