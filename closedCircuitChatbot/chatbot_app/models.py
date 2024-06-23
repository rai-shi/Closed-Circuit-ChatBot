from django.db import models
from django.contrib.auth.models import User
    

class Chats(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"chat id: {self.id}"
    

class Prompts(models.Model):
    # id = models.AutoField(auto_created=True, primary_key=True, serialize=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    chat_id = models.ForeignKey(Chats, on_delete=models.CASCADE)
    question = models.CharField(max_length=10000)
    response = models.CharField(max_length=10000)

    def __str__(self):
        return f"question: {self.question} response: {self.response}"

