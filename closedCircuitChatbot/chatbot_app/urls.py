from django.urls import path, re_path
from . import views

app_name = 'chatbot_app'

urlpatterns = [
    path("", views.home), 
	# path("home", views.home, name="home"), 
    re_path(r'^home/(?P<chat_id>-?\d+)/$', views.home, name='home'),
    re_path(r'^home/handlePrompt/(?P<chat_id>-?\d+)/$', views.handlePrompt, name='handlePrompt'),
	# path("home/handlePrompt", views.handlePrompt, name="handlePrompt"), 
	path("home/deneme", views.deneme, name="deneme"), 
    path("home/CreateChat", views.CreateChat, name="CreateChat"),  
    re_path(r"^home/DeleteChat/(?P<current_chat_id>-?\d+)/(?P<delete_chat_id>-?\d+)/$", views.DeleteChat, name="DeleteChat"), 
]

