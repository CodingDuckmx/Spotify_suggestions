from django import forms

class AddingUsername(forms.Form):
    sp_username = forms.CharField(label='Enter your Spotify username',max_length=100)

class SongsName(forms.Form):
    entered_song_name = forms.CharField(label='Enter a song name',max_length=100)
    entered_artists = forms.CharField(label='Enter the artist name',max_length=100)