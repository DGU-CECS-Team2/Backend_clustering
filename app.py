from flask import Flask, request, jsonify
from clustering.data_preparation import DataPreparer
from clustering.pca import PCAReducer
from clustering.dbscan import DBSCANClusterer
import numpy as np
from flask_cors import CORS
from json import JSONEncoder as BaseJSONEncoder
from sklearn.preprocessing import MinMaxScaler
from clustering.tsne import TSNEReducer
from sklearn.manifold import TSNE

class JSONEncoder(BaseJSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return super(JSONEncoder, self).default(obj)

app = Flask(__name__)
CORS(app)
app.json_encoder = JSONEncoder

@app.route('/cluster-data', methods=['POST'])
def cluster_data():
    # POST 요청으로부터 데이터를 받습니다.
    received_data = request.json['data']
    
    # `member_id`와 나머지 데이터를 분리합니다.
    processed_data = []
    member_ids = []
    for item in received_data:
        member_id = item['member_id']
        member_ids.append(member_id)
        processed_item = [item['age'], item['gender'], item['x_coord'], item['y_coord'], item['tv'], item['movie'], item['reading']]
        processed_data.append(processed_item)
        
    # DataPreparer 클래스의 인스턴스를 생성하여 데이터를 준비합니다.
    # preparer = DataPreparer()
    # summarized_data = preparer.summarize_interests(np.array(processed_data))
    # scaled_data = preparer.scale_data(summarized_data)
    
    # 데이터 스케일링
    scaler = DataPreparer()
    scaled_data = scaler.scale_data(np.array(processed_data))
    
    # PCA로 차원 축소를 수행합니다.
    # reducer = PCAReducer(n_components=2)
    # reduced_data = reducer.reduce(scaled_data)
    
    
    # t-SNE를 사용하여 차원 축소
    tsne_reducer = TSNEReducer(n_components=2, random_state=42)
    reduced_data = tsne_reducer.reduce(scaled_data)

    
    # DBSCAN으로 클러스터링합니다.
    clusterer = DBSCANClusterer(eps=0.3, min_samples=5)
    clusters = clusterer.fit_predict(reduced_data)
    group_ids = clusterer.group_data_within_clusters(reduced_data, clusters, max_group_size=7)

   # 클러스터 결과를 구조화하여 반환합니다.
    results = [{'group_id': group_id, 'member_ids': [member_ids[i] for i, gid in enumerate(group_ids) if gid == group_id]} for group_id in range(len(set(group_ids)))]


    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
