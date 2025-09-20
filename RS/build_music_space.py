# build_music_space_colab.py
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from config import spotify_config
from urllib.parse import urlparse, parse_qs

# -----------------------------
# Step 1: Authenticate Spotify
# -----------------------------
sp_oauth = SpotifyOAuth(
    client_id=spotify_config["client_id"],
    client_secret=spotify_config["client_secret"],
    redirect_uri=spotify_config["redirect_uri"],
    scope="user-library-read playlist-read-private"
)

auth_url = sp_oauth.get_authorize_url()
print(f"1) Go to this URL in your browser:\n{auth_url}\n")
print("2) Log in and authorize, then copy the URL you are redirected to.\n")

redirect_response = input("Enter the full redirected URL: ").strip()
code = parse_qs(urlparse(redirect_response).query).get("code")
if not code:
    raise ValueError("Failed to get code from redirected URL.")
code = code[0]

token_info = sp_oauth.get_access_token(code)
sp = spotipy.Spotify(auth=token_info["access_token"])

# -----------------------------
# Step 2: Fetch playlist tracks
# -----------------------------
tracks = []
results = sp.playlist_items(spotify_config["playlist_id"], additional_types=["track"])
while results:
    for item in results["items"]:
        track = item["track"]
        if track:
            tracks.append({
                "id": track["id"],
                "name": track["name"],
                "artist": ", ".join([a["name"] for a in track["artists"]]),
                "release_year": track["album"]["release_date"][:4],
                "genre": None
            })
    if results["next"]:
        results = sp.next(results)
    else:
        results = None

# -----------------------------
# Step 3: Fetch audio features in batches
# -----------------------------
def fetch_audio_features(sp, track_ids):
    all_features = []
    batch_size = 100  # Spotify API limit
    for i in range(0, len(track_ids), batch_size):
        batch = track_ids[i:i+batch_size]
        features = sp.audio_features(batch)
        all_features.extend(features)
    return all_features

ids = [t["id"] for t in tracks]
features = fetch_audio_features(sp, ids)

df_features = pd.DataFrame(features)[["id", "valence", "energy", "danceability"]]
df_tracks = pd.DataFrame(tracks)

# Merge metadata + audio features
df = pd.merge(df_tracks, df_features, on="id", how="inner")

# -----------------------------
# Step 4: Optional: Fetch artist genres
# -----------------------------
for i, row in df.iterrows():
    search_results = sp.search(q=row["artist"], type="artist")["artists"]["items"]
    if search_results:
        artist_id = search_results[0]["id"]
        genres = sp.artist(artist_id)["genres"]
        df.at[i, "genre"] = genres[0] if genres else "Unknown"
    else:
        df.at[i, "genre"] = "Unknown"

# -----------------------------
# Step 5: Save final CSV
# -----------------------------
df.to_csv(spotify_config["spotify_space_csv"], index=False)
print(f"[INFO] Saved Spotify vector space with metadata to {spotify_config['spotify_space_csv']}")
