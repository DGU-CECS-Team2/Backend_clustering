import numpy as np
import random

def generate_data(num_samples=100):
    data = []
    for i in range(1, num_samples + 1):
        user_id = f"user{i}"
        age = random.randint(60, 90)
        gender = random.randint(0, 1)
        x_coord = round(random.uniform(126, 130), 2)
        y_coord = round(random.uniform(33, 38), 2)
        tv = random.randint(40, 90)
        movie = random.randint(40, 90)
        reading = random.randint(40, 90)
        data.append({"member_id": user_id, "age": age, "gender": gender, "x_coord": x_coord, "y_coord": y_coord, "tv": tv, "movie": movie, "reading": reading})
    return data

if __name__ == "__main__":
    generated_data = generate_data()
    for row in generated_data:
        print(row)
