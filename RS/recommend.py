import pandas as pd
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import spotify_config
import os

# -------------------------------
# Load emotion inference
# -------------------------------
with open("/home/ml/Desktop/shubham/sensors/RS/outputs/gpu_inference_fp32.json", "r") as f:
    emotion_data = json.load(f)

valence, arousal = emotion_data["valence"], emotion_data["arousal"]

# -------------------------------
# Load music space
# -------------------------------
df = pd.read_csv("/home/ml/Desktop/shubham/sensors/RS/Spotify_changed.csv")
for col in ["track_id", "track_name", "artists", "album", "release_year", "mood"]:
    if col not in df.columns:
        raise ValueError(f"Column {col} not found in CSV!")

# Map mood -> valence & arousal
mood_to_valence_arousal = {
    "happy": (0.9, 0.8),
    "sad": (0.2, 0.3),
    "relaxed": (0.7, 0.2),
    "angry": (0.1, 0.9),
    "neutral": (0.5, 0.5)
}

df["valence"] = df["mood"].map(lambda x: mood_to_valence_arousal.get(x, (0.5,0.5))[0])
df["energy"] = df["mood"].map(lambda x: mood_to_valence_arousal.get(x, (0.5,0.5))[1])
df["danceability"] = np.random.uniform(0.4, 0.9, size=len(df))

# -------------------------------
# Load or create user history
# -------------------------------
user_history_path = spotify_config["user_history_csv"]

if os.path.exists(user_history_path):
    user_history = pd.read_csv(user_history_path)
else:
    # Initialize with all tracks as not liked
    user_history = pd.DataFrame({
        "track_id": df["track_id"],
        "liked": 0
    })
    user_history.to_csv(user_history_path, index=False)
    print("[INFO] Created new user_history.csv")

# Merge to get 'liked' column
df = pd.merge(df, user_history, left_on="track_id", right_on="track_id", how="left")
df["liked"] = df["liked"].fillna(0)

# -------------------------------
# Function to compute recommendation scores
# -------------------------------
def compute_scores(df, valence, arousal):
    df["distance"] = ((df["valence"] - valence) ** 2 + (df["energy"] - arousal) ** 2) ** 0.5
    df.loc[df["liked"] == 1, "distance"] *= 0.5
    df["novelty"] = np.where(df["liked"] == 0, 1.0, 0.2)

    features = df[["valence", "energy", "danceability"]].values
    sim_matrix = cosine_similarity(features)
    diversity_scores = []
    for i in range(len(df)):
        sims = np.sort(sim_matrix[i])[-6:-1]
        diversity_scores.append(1 - np.mean(sims))
    df["diversity"] = diversity_scores

    # Normalize
    df["distance_norm"] = (df["distance"] - df["distance"].min()) / (df["distance"].max() - df["distance"].min())
    df["novelty_norm"] = (df["novelty"] - df["novelty"].min()) / (df["novelty"].max() - df["novelty"].min())
    df["diversity_norm"] = (df["diversity"] - df["diversity"].min()) / (df["diversity"].max() - df["diversity"].min())

    w_distance, w_novelty, w_diversity = 0.5, 0.25, 0.25
    df["final_score"] = (
        w_distance * df["distance_norm"] -
        w_novelty * df["novelty_norm"] -
        w_diversity * df["diversity_norm"]
    )
    return df

# -------------------------------
# Initial recommendation
# -------------------------------
df = compute_scores(df, valence, arousal)
top_k = spotify_config.get("top_k", 10)
recommendations = df.sort_values("final_score").head(top_k)

print("\n--- Top Recommendations ---")
print(recommendations[["track_id", "track_name", "artists", "liked"]])

# -------------------------------
# Ask user to like/dislike
# -------------------------------
print("\nEnter track_ids you want to like (comma-separated):")
liked_ids = input().strip().split(",")
liked_ids = [x.strip() for x in liked_ids]

print("\nEnter track_ids you want to dislike (comma-separated):")
disliked_ids = input().strip().split(",")
disliked_ids = [x.strip() for x in disliked_ids]

# Update user history
user_history.loc[user_history["track_id"].isin(liked_ids), "liked"] = 1
user_history.loc[user_history["track_id"].isin(disliked_ids), "liked"] = 0
user_history.to_csv(user_history_path, index=False)
print("[INFO] user_history.csv updated.")

# -------------------------------
# Update user library (only liked songs)
# -------------------------------
library_path = spotify_config.get("user_library_csv", "RS/outputs/user_library.csv")

# Keep only liked songs
liked_songs = user_history[user_history["liked"] == 1].merge(
    df[["track_id", "track_name", "mood"]],
    on="track_id",
    how="left"
)

# Select only required columns
user_library = liked_songs[["track_name", "mood"]]

# Save library
os.makedirs(os.path.dirname(library_path), exist_ok=True)
user_library.to_csv(library_path, index=False)
print(f"[INFO] User library (liked songs only) saved to {library_path}")



# -------------------------------
# Recompute recommendations after feedback
# -------------------------------
df = pd.merge(df.drop(columns=["liked"]), user_history, on="track_id", how="left")
df["liked"] = df["liked"].fillna(0)
df = compute_scores(df, valence, arousal)

# Get top 7 new songs (not liked yet)
top_new = df[df["liked"] == 0].sort_values("final_score").head(7)

# Get last 3 liked songs
last_liked = df[df["liked"] == 1].sort_values("final_score").tail(3)

final_recs = pd.concat([top_new, last_liked])
print("\n--- Final Recommendations (Top 7 + Last 3 liked) ---")
print(final_recs[["track_id", "track_name", "artists", "liked", "final_score"]])

# Save final recommendations
recommendations_csv = spotify_config.get("recommendations_csv", "outputs/final_recommendations.csv")
os.makedirs(os.path.dirname(recommendations_csv), exist_ok=True)
final_recs.to_csv(recommendations_csv, index=False)
print(f"[INFO] Final recommendations saved to {recommendations_csv}")
