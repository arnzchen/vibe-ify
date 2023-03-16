import requests
import base64
import datetime


client_id = '9da2ab8acd9d404c9867727901a6c9e4'; 
client_secret = 'b91cf2e764a044f69b345599a44f2b37'; 
redirect_url = 'vibe-ify-login://callback'; 

class SpotifyAPIClient(object):
    bearer_token = None
    bearer_token_expiry = datetime.datetime.now()
    token_usage_delta = datetime.timedelta(milliseconds=10)
    client_id = None
    client_secret = None
    redirect_url = None
    api_retries = 3

    def __init__(self, client_id, client_secret, redirect_url):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_url = redirect_url
        
    def authenticate(self):
        client_creds = f"{self.client_id}:{self.client_secret}"
        creds_b64 = base64.b64encode(client_creds.encode()).decode()

        auth_url = "https://accounts.spotify.com/api/token"
        auth_req_header = {'Authorization': 'Basic ' + creds_b64, 'Content-Type': 'application/x-www-form-urlencoded'}
        auth_req_data = {'grant_type': 'client_credentials', 'redirect_uri': self.redirect_url}
        auth_res = requests.post(auth_url, headers=auth_req_header, data=auth_req_data)

        if auth_res.status_code != 200:
            raise Exception("Client Auth Failed.")

        json_res = auth_res.json()
        expires_in = json_res['expires_in']
        now = datetime.datetime.now()
        self.bearer_token_expiry = now + datetime.timedelta(seconds=expires_in)
        self.bearer_token = json_res['access_token']
    
    def get_bearer_token(self):
        now = datetime.datetime.now()
        if self.bearer_token is None or self.bearer_token_expiry < now - self.token_usage_delta:
            self.authenticate()
            return self.get_bearer_token()
        return self.bearer_token

    def concat_url_params(self, url, params):
        concat_url = url
        for key, value in params.items():
            concat_url += f'&{key}={value}'
        return concat_url
    
    def get_available_genres(self): 
        bearer_token = self.get_bearer_token()
        genres_endpoint = "https://api.spotify.com/v1/recommendations/available-genre-seeds"
        all_genres = []

        genres_req_headers = {"Content-type": "application/json", "Authorization": f"Bearer {bearer_token}"}

        response = requests.get(genres_endpoint, headers=genres_req_headers)

        if response:
            print("Available Genres:")
            json_response = response.json()
            print(json_response)
        else:
            raise Exception("Failed to call available genre seeds endpoint", response)



    def get_reccomended_songs(self, limit=5, seed_artists='', seed_tracks='0c6xIDDpzE81m2q797ordA', market="US",
                              seed_genres='', target_danceability=0.1): 
        bearer_token = self.get_bearer_token()
        recs_endpoint = "https://api.spotify.com/v1/recommendations?"
        all_recs = []

        recs_url = self.concat_url_params(recs_endpoint, 
            {'limit': limit, 'market': market, 'seed_genres': seed_genres, 'target_danceability': target_danceability, 
            'seed_artists': seed_artists, 'seed_tracks': seed_tracks})
        recs_req_headers = {"Content-type": "application/json", "Authorization": f"Bearer {bearer_token}"}
        response = requests.get(recs_url, headers=recs_req_headers)
        if response:
            print("Reccomended songs:")
            json_response = response.json()
            for i, j in enumerate(json_response['tracks']):
                track_name = j['name']
                artist_name = j['artists'][0]['name']
                link = j['artists'][0]['external_urls']['spotify']

                print(f"{i+1}) \"{j['name']}\" by {j['artists'][0]['name']}")
                reccs = [track_name, artist_name, link]
                all_recs.append(reccs)
            return all_recs
        else:
            raise Exception("Failed to call song recommendation endpoint", response)


# Testing 
client = SpotifyAPIClient(client_id=client_id, client_secret=client_secret, redirect_url=redirect_url)
# client.get_available_genres()
client.get_reccomended_songs()
