import requests

# https://developer.spotify.com/documentation/web-api/reference/#/operations/get-recommendations
response = requests.get("https://api.spotify.com/v1/recommendations")