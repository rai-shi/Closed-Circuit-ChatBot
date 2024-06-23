
from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views 
from django.conf import settings
from django.conf.urls.static import static
from user_app import views as user_app
from chatbot_app import views as chatbot_app

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('user_app.urls')), # user veri tabanı işlemi karar verilene kadar kayıt-giriş sayfası kapatılmıştır. 
    # path('', include('chatbot_app.urls')), # kullanıcılar doğrudan anasayfaya yönlendirilecektir.
    path("user/", include('user_app.urls')),
    path("botai/", include('chatbot_app.urls')),
    path("botaiAdmin/", include('admin_app.urls')),
]
