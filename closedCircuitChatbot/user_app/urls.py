from django.urls import path
from . import views
from chatbot_app import views as chatbot_views

# app_name = 'user_app'


urlpatterns = [
	path("", views.index, name="index"), #herhangi bir sayfaya geçmeden kaynaktan gönderim 
	path("index", views.index),
	path("UserError", views.UserError, name="UserError"),
	path("Login", views.Login, name="Login"),
	path("Register", views.Register, name="Register"),  
	path("Profile", views.Profile, name="Profile"),  
	path("Logout", views.Logout, name="Logout"),  
	path("UpdateProfile", views.UpdateProfile, name="UpdateProfile"),  
]

