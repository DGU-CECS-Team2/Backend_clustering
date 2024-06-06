from flask import Flask, request, jsonify
import numpy as np
import random
import json
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from flask_cors import CORS
from clustering.genetic_algorithm import GeneticAlgorithmForClustering

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return super(JSONEncoder, self).default(obj)

app = Flask(__name__)
CORS(app)
app.json_encoder = JSONEncoder

# Flask의 기본 로거 설정
logging.basicConfig(level=logging.DEBUG)
app.logger.addHandler(logging.StreamHandler())
app.logger.setLevel(logging.DEBUG)

@app.route('/cluster-data', methods=['POST'])
def cluster_data():
    app.logger.info("클러스터링 데이터 요청 받음")
    
    try:
        received_data = request.json['data']
        app.logger.debug(f"받은 데이터: {received_data}")

        # `member_id`와 나머지 데이터를 분리합니다.
        processed_data = []
        member_ids = []
        for item in received_data:
            member_id = item['member_id']
            member_ids.append(member_id)
            processed_item = [item['age'], item['gender'], item['interest_1'], item['interest_2'], item['interest_3'], item['interest_4'], item['interest_5']]
            processed_data.append(processed_item)
        
        app.logger.debug(f"처리된 데이터: {processed_data}")
        app.logger.debug(f"멤버 IDs: {member_ids}")
        
        # 데이터 스케일링
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(np.array(processed_data))
        print("스케일링")
        for row in scaled_data:
            print("[", ", ".join(f"{x:.2f}" for x in row), "],")
        # app.logger.debug(f"스케일링된 데이터: {scaled_data}")
    
        
        # t-SNE를 사용하여 차원 축소
        tsne_reducer = TSNE(n_components=2, random_state=42)
        reduced_data = tsne_reducer.fit_transform(scaled_data)
        app.logger.debug(f"차원 축소 결과: {reduced_data}")
        
        # 첫 번째 항에 member_ids 추가함
        combined_data = [[member_id, *coords] for member_id, coords in zip(member_ids, reduced_data)]
        app.logger.debug(f"조합된 데이터: {combined_data}")
        
                        # NearestNeighbors를 사용해 k-거리 계산
        def estimate_eps(data):
            neighbors = NearestNeighbors(n_neighbors=4)
            neighbors_fit = neighbors.fit(data)
            distances, indices = neighbors_fit.kneighbors(data)
            distances = np.sort(distances, axis=0)[:, 3]
            return np.percentile(distances, 90)  # 90번째 백분위수를 사용
        
        estimated_eps = estimate_eps(reduced_data)
        app.logger.debug(f"추정된 EPS: {estimated_eps}")

        # 클러스터링
        genetic_clustering = GeneticAlgorithmForClustering(combined_data, estimated_eps)
        results = genetic_clustering.run_algorithm();
        print(results)
        
        #모든 group_id를 Python의 int로 변환
        for result in results:
            result['group_id'] = int(result['group_id'])
        app.logger.info(f"최종 클러스터링 결과: {results}")
        
        return jsonify(results)
    
    except Exception as e:
        app.logger.error(f"클러스터링 중 오류 발생: {e}")
        return jsonify({'error': '클러스터링 처리 중 오류 발생'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
