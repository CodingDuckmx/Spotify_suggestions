from .scripts_and_pickes.spotbot import SpotBot
from .forms import AddingUsername, SongsName
from .models import Post
from django.shortcuts import render


sb = SpotBot()

def home(request):
    context = {
        # 'posts': Post.objects.all()
    }
    # If this is a POST request we need to process the form data
    if request.method == 'POST':
        # Create a form instance and populate it with the data from the request
        form = SongsName(request.POST)
        if form.is_valid():
            # process the data in form.cleaned_data as requerided

            entered_song_name = form.cleaned_data['entered_song_name']
            entered_artists = form.cleaned_data['entered_artists']

            songs_similarities_list = sb.pull_similar_songs_from_db(entered_song_name, entered_artists)

            context['song_name'] = entered_song_name
            context['similar_songs'] = songs_similarities_list

            if songs_similarities_list:

                context['songs_urls'] = []

                for song_id in songs_similarities_list:

                    context['songs_urls'].append('https://open.spotify.com/embed/track/{}'.format(song_id))

            context['searched'] = True
            context.update({'form':form})
            return render(request, 'song_matcher/home.html', context) 

    else:

        form = SongsName()
        context['song_name'] = None
        context.update({'form':form})
        return render(request, 'song_matcher/home.html', context) 

def about(request):
    return render(request, 'song_matcher/about.html', {'title': 'About'})



def contribute(request):

    context = {
        'posts': Post.objects.all()
    }
    # If this is a POST request we need to process the form data
    if request.method == 'POST':
        # Create a form instance and populate it with the data from the request
        form = AddingUsername(request.POST)
        if form.is_valid():
            # process the data in form.cleaned_data as requerided

            sp_username = form.cleaned_data['sp_username']

            sb.store_songs(username=sp_username)

            context['username'] = sp_username
            context.update({'form':form})
            return render(request, 'song_matcher/contribute.html', context) 

    else:

        form = AddingUsername()
        context['username'] = None
        context.update({'form':form})
        return render(request, 'song_matcher/contribute.html', context) 

def recalibrate(request):

    sb.update_similarities(similarity_threshold = 0.84)

    sb.update_db_similarities_from_pickles()

    return render(request, 'song_matcher/recalibrated.html') 
