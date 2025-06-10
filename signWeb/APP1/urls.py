from django.conf.urls.static import static
from django.urls import path, include
from . import views
from django.conf import settings

#from APP1.views import fetch_video_path, index
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('', include('admin_adminlte.urls')),
    path('index.html', views.home, name='home'),
    path('service.html', views.service, name='service'),
    path('about.html', views.about, name='about'),
    path('contact.html', views.contact, name='contact'),
    path('signin.html', views.Login, name='signin'),
    path('signup.html', views.signup, name='signup'),
    path('Logout', views.user_logout, name='Logout'),
    path('CommentForm', views.commentform, name='CommentForm'),
    path('signToVoice.html', views.signToVoice, name='signToVoice'),
    path('TextToSign.html', views.TextToSign, name='TextToSign'),
    path('VoiceToSign.html', views.VoiceToSign, name='VoiceToSign'),
    path('signToText.html', views.signToText, name='signToText'),

    path('signToText.html', views.index, name='index'),
    path('TextToSign.html', views.index_word, name='index_word'),
    path('SignToVoice.html', views.index_numbers, name='index_numbers'),

    path('video_feed/', views.video_feed, name='video_feed'),
    path('video_feed_word/', views.video_feed_word, name='video_feed_word'),
    path('video_feed_numbers/', views.video_feed_numbers, name='video_feed_numbers'),
    path('fetch_predicted_sentence/', views.fetch_predicted_sentence, name='fetch_predicted_sentence'),
    path('add_space/', views.add_space, name='add_space'),
    path('delete_letter/', views.delete_letter, name='delete_letter'),

    path('VoiceToSign.html', views.index_sign, name='index_sign'),
    path('fetch_video_path/', views.fetch_video_path, name='fetch_video_path'),

]
# Add this to serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

