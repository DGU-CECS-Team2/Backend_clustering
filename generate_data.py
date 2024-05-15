import numpy as np
import random
import json

def generate_data(num_samples=100):
    data = []
    for i in range(1, num_samples + 1):
        user_id = i
        age = random.randint(60, 90)
        gender = random.randint(0, 1)
        interest_1 = random.randint(0, 100)
        interest_2 = random.randint(0, 100)
        interest_3 = random.randint(0, 100)
        interest_4 = random.randint(0, 100)
        interest_5 = random.randint(0, 100)
        data.append({
            "member_id": i,
            "age": age,
            "gender": gender,
            "interest_1": interest_1,
            "interest_2": interest_2,
            "interest_3": interest_3,
            "interest_4": interest_4,
            "interest_5": interest_5
        })
    return data

if __name__ == "__main__":
    generated_data = generate_data()
    # JSON 형식으로 출력 (data 키 아래에 데이터 포함)
    output = {
        "data": generated_data
    }
    print(json.dumps(output, ensure_ascii=False, indent=4))
