from . import views
from django.urls import path

urlpatterns = [
    path('', views.home, name = 'sm-home'),
    path('about/', views.about, name = 'sm-about'),
]