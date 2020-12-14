from . import views
from django.urls import path

urlpatterns = [
    path('', views.home, name = 'sm-home'),
    path('contribute/', views.contribute, name = 'sm-contribute'),
    path('about/', views.about, name = 'sm-about'),
    path('recalibrate/', views.recalibrate, name = 'sm-recalibrate'),
]