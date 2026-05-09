# file: play_by_emotion.py
# pip install spotipy flask
import os
from flask import Flask, request, redirect
import spotipy
from spotipy.oauth2 import SpotifyOAuth

CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID", "f66ced5a6d7c4b9fac274fa25a7df9d8")
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "a1fd7b07e77b455aa232e001a18f724d")
REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
SCOPE = "user-modify-playback-state user-read-playback-state user-read-currently-playing playlist-read-private"

# mapping emotion -> playlist URI (thay bằng playlist bạn muốn)
EMOTION_PLAYLISTS = {
    "happy": "spotify:playlist:37i9dQZF1DXdPec7aLTmlC",
    "sad":   "spotify:playlist:37i9dQZF1DX7qK8ma5wgG1",
    "neutral":  "spotify:playlist:37i9dQZF1DX4sWSpwq3LiO",
}

app = Flask(__name__)
sp_oauth = SpotifyOAuth(client_id=CLIENT_ID,
                        client_secret=CLIENT_SECRET,
                        redirect_uri=REDIRECT_URI,
                        scope=SCOPE,
                        cache_path=".cache-playbyemotion")  # token cache

@app.route("/")
def index():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route("/callback")
def callback():
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)
    access_token = token_info['access_token']
    # tạo client
    sp = spotipy.Spotify(auth=access_token)
    return "Authorized! Now call /play/<emotion> to play."

@app.route("/play/<emotion>")
def play_emotion(emotion):
    token_info = sp_oauth.get_cached_token()
    if not token_info:
        return redirect("/")  # chưa authorized
    sp = spotipy.Spotify(auth=token_info['access_token'])

    # kiểm tra thiết bị hoạt động
    devices = sp.devices().get('devices', [])
    if not devices:
        return "No active devices found. Open Spotify (desktop/mobile) and make it active.", 400

    playlist = EMOTION_PLAYLISTS.get(emotion.lower())
    if not playlist:
        return f"Unknown emotion: {emotion}", 400

    # bắt đầu phát playlist trên thiết bị active (Spotify cho biết user phải có Premium)
    sp.start_playback(context_uri=playlist)  # hoặc sp.start_playback(device_id=device_id, context_uri=playlist)
    return f"Started playing {emotion}"

if __name__ == "__main__":
    app.run(port=8888, debug=True)
