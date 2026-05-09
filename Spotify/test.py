import spotipy
from spotipy.oauth2 import SpotifyOAuth

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="f66ced5a6d7c4b9fac274fa25a7df9d8",
    client_secret="a1fd7b07e77b455aa232e001a18f724d",
    redirect_uri="http://127.0.0.1:8888/callback",
    scope="user-read-playback-state"
))

me = sp.me()
print("Xin chào,", me["display_name"])
