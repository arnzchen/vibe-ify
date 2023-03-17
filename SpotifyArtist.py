class SpotifyArtist(object):
    name = None
    id = None

    def __init__(self, name, id):
        self.name = name
        self.id = id
    
def extract_from_json(json):
    artist_name = json['name']
    artist_id = json['id']
    return SpotifyArtist(artist_name, artist_id)


