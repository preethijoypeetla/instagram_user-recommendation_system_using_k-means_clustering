from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load trained model
model = pickle.load(open("kmeans_model.pkl","rb"))

@app.get("/")
def home():
    return {"message": "Instagram User Recommendation System"}

@app.post("/predict")
def predict(follower_count:int, likes:int, comments:int, shares:int, saves:int, reach:int, impressions:int, engagement_rate:float):

    data = np.array([[follower_count,likes,comments,shares,saves,reach,impressions,engagement_rate]])

    cluster = model.predict(data)

    return {"recommended_cluster": int(cluster[0])}