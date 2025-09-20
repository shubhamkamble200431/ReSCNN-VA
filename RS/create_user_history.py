import pandas as pd

# Load your main music dataset
spotify_df = pd.read_csv("RS/Spotify_changed.csv")

# Create a blank user history
user_history = pd.DataFrame({
    "track_id": spotify_df["track_id"],
    "liked": 0  # default: no likes yet
})

# Save template
user_history.to_csv("RS/user_history.csv", index=False)
print("[INFO] User history template created: user_history.csv")
