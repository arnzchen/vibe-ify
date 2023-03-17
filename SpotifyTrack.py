import SpotifyArtist

class SpotifyTrack(object):
    name = None
    id = None
    artists = None

    def __init__(self, name, id, artists):
        self.name = name
        self.id = id
        self.artists = artists
    
def extract_from_json(json):
    track_name = json['name']
    track_id = json['id']
    track_artists = [SpotifyArtist.extract_from_json(artist_json) for artist_json in json['artists']]
    return SpotifyTrack(track_name, track_id, track_artists)

