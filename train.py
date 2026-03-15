import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
data = pd.read_csv("Instagram_Analytics.csv")

# Select important numeric features for clustering
features = data[['follower_count','likes','comments','shares','saves','reach','impressions','engagement_rate']]

# Remove missing values
features = features.dropna()

# Scale the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Train K-Means model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)

# Save trained model
with open("kmeans_model.pkl","wb") as f:
    pickle.dump(kmeans,f)

print("Model trained successfully!")