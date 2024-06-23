from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=200, blank=False, null=False)
    unit = models.CharField(max_length=200, blank=False, null=False)

    def __str__(self):
        return self.user.username
    
