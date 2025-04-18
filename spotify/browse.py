import os
from dotenv import load_dotenv, find_dotenv
import base64
from requests import post, get
import json

load_dotenv(find_dotenv())

client_id = os.getenv("SPOTIFY_API_CLIENT")
client_secret = os.getenv("SPOTIFY_API_SECRET")


def get_token():
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    result = post(url, headers=headers, data=data)
    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token


def get_auth_header():
    return {"Authorization": "Bearer " + get_token()}


def search_for_artist(artist_name):
    url = "https://api.spotify.com/v1/search"
    headers = get_auth_header()
    query = f"?q={artist_name}&type=artist&limit=1"
    query_url = url + query
    result = get(query_url, headers=headers)
    json_result = json.loads(result.content)["artists"]["items"]
    if len(json_result) == 0:
        print("No artist with this name exists...")
        return None
    return json_result[0]


def search_for_playlist(playlist_name):
    url = "https://api.spotify.com/v1/search"
    headers = get_auth_header()
    query = f"?q={playlist_name}&type=playlist&limit=1"
    query_url = url + query
    result = get(query_url, headers=headers)
    json_result = json.loads(result.content)["playlists"]["items"]
    if len(json_result) == 0:
        print("No playlists with this name exists...")
        return None
    return json_result[0]


def get_top_songs_by_playlist(playlist_id):
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}"
    headers = get_auth_header()
    result = get(url, headers=headers)
    json_result = json.loads(result.content)["tracks"]
    return json_result


def get_top_songs_by_artist(artist_id):
    url = f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks?country=MX"
    headers = get_auth_header()
    result = get(url, headers=headers)
    json_result = json.loads(result.content)["tracks"]
    return json_result


def top_10_songs_by_artist_name(artist_name):
    artist_id = search_for_artist(artist_name)["id"]
    top_songs = get_top_songs_by_artist(artist_id)
    top_10_songs = {}
    for i, song in enumerate(top_songs):
        print(f" {i + 1} - {song['name']} ({song['popularity']}/100)")
        top_10_songs[song['name']] = song['popularity']
    return top_10_songs


def top_10_songs_by_playlist(playlist_name):
    playlist_id = search_for_playlist(playlist_name)["id"]
    top_songs = get_top_songs_by_playlist(playlist_id)["items"]
    top_10_songs = {}
    for i, song in enumerate(top_songs):
        print(
            f" {i + 1} {song['track']['artists'][0]['name']} - {song['track']['name']} ({song['track']['popularity']}/100)")
        top_10_songs[f"{song['track']['artists'][0]['name']} - {song['track']['name']}"] = song['track']['popularity']
        if i == 9:
            break
    return top_10_songs


#top_10 = top_10_songs_by_playlist("Trash Metal")

#print(max(top_10, key=lambda key: top_10[key]))
