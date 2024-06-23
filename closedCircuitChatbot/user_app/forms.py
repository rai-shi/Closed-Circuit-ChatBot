from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.core.exceptions import ValidationError
from django.contrib.auth.validators import UnicodeUsernameValidator
from django.contrib.auth import password_validation
from .models import UserProfile

from pymongo import MongoClient
client = MongoClient()


def validate_email(value):
    users = User.objects.filter(email=value)
    if users:
        raise forms.ValidationError(f"{value} is already taken. Please choose a different one.", params = {'value':value})
    return value

def validate_username(value):
    users = User.objects.filter(username=value)
    if users:
        raise forms.ValidationError(f"{value} is already taken. Please choose a different one.", params = {'value':value})
    return value
    

class SignupForm(UserCreationForm):

    first_name = forms.CharField(
                            max_length=25, 
                            min_length=4, 
                            required=True, 
                            help_text='Required: First Name',
                            widget=forms.TextInput(attrs={'class': 'input-field', 'placeholder': 'Enter first name'}))
    
    last_name = forms.CharField(
                            max_length=25, 
                            min_length=2, 
                            required=True, 
                            help_text='Required: Last Name',
                            widget=(forms.TextInput(attrs={'class': 'input-field', 'placeholder': 'Enter last name'})))

    email = forms.EmailField(
                            max_length=200, 
                            required=True, 
                            help_text='Required: Email', 
                            validators = [validate_email],
                            widget=(forms.EmailInput(attrs={'class': 'input-field', 'placeholder': 'Enter your email address'})))
    
    password1 = forms.CharField(
        widget=(forms.PasswordInput(attrs={'class': 'input-field', 'placeholder': 'Enter password'})),
        help_text=password_validation.password_validators_help_text_html())
    
    password2 = forms.CharField(
        widget=(forms.PasswordInput(attrs={'class': 'input-field', 'placeholder': 'Validate password'})),
        help_text=password_validation.password_validators_help_text_html())
    
    title = forms.CharField(
        max_length=200,
        required=True,
        help_text='Required: Title',
        widget=forms.TextInput(attrs={'class': 'input-field', 'placeholder': 'Enter your job title'})
    )
    unit = forms.CharField(
        max_length=200,
        required=True,
        help_text='Required: Unit',
        widget=forms.TextInput(attrs={'class': 'input-field', 'placeholder': 'Enter your working unit'})
    )

    # not used in frontend, username is created by combining first_name and last_name in the background 
    # username = first_name + "_" + last_name
    username = forms.CharField(
        max_length=50,
        min_length=5, 
        required=True,
        help_text=('Required. 50 characters or fewer, more than 5 characters. Letters, digits and @/./+/-/_ only.'),
        validators=[validate_username],
        error_messages={'unique': ("A user with that username already exists.")},
        widget=forms.TextInput(attrs={'class': 'input-field', 'placeholder': 'Enter username'})
    )

    

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2', 'title', 'unit')

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
            user_profile = UserProfile.objects.create(
                user=user,
                title=self.cleaned_data['title'],
                unit=self.cleaned_data['unit']
            )
        return user
    
    
