from .models import Post
from django.shortcuts import render


def home(request):
    context = {
        'posts': Post.objects.all()
    }
    return render(request, 'song_matcher/home.html', context)

def about(request):
    return render(request, 'song_matcher/about.html', {'title': 'About'})