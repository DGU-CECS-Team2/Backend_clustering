import requests
from generate_data import generate_data

# 데이터 생성
generated_data = generate_data(num_samples=100)

# 서버에 데이터 전송
url = 'http://127.0.0.1:5000/cluster-data'
response = requests.post(url, json={"data": generated_data})

# 응답 확인
print(response.text)
