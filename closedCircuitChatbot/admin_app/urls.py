from django.urls import path, re_path
from . import views

from django.conf import settings
from django.conf.urls.static import static

app_name = 'admin_app'

urlpatterns = [
    path("", views.AdminMain), 
    path("admin/", views.AdminMain, name="AdminMain"),  
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
