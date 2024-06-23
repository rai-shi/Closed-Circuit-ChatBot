from django.shortcuts import render, redirect, HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash

from . import authentication
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from .forms import SignupForm
from django.contrib.sites.shortcuts import get_current_site  
from django.utils.encoding import force_bytes, force_str  
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode  
from django.template.loader import render_to_string  
from .tokens import account_activation_token  
from django.contrib.auth.models import User  
from django.contrib.auth import get_user_model
from django.core.mail import EmailMessage  
from django.apps import apps
from chatbot_app.models import Chats


from .models import UserProfile

from django.contrib.auth.models import Group 

from django.contrib import messages
# Create your views here.

def index(request):
    form = SignupForm()  
    return render(request, 'login_register.html', {'form': form} )

def UserError(request):
    form = SignupForm()  
    return render(request, 'login_register.html', {'form': form} )

def Login(request):
    if request.method == 'POST':  

        form_type = request.POST.get('form_type', None)

        if form_type == 'Login':
            email = request.POST.get('email')
            password = request.POST.get('password')

            
            user = authenticate(request, email=email, password=password)
            last_chat = Chats.objects.filter(user=user).order_by('-updated_at').first()
            if last_chat:
                chat_id = last_chat.id
            else:
                chat_id = -1

            if user is not None:
                login(request, user)
                home_url = reverse('chatbot_app:home', kwargs={"chat_id":chat_id})
                return redirect(home_url)
            else:
                messages.error(request, message = "Email or password is wrong. Try again...")
                return redirect("index")
        else:
            messages.error(request, "Form type is not as expected.")
            return redirect("index")
        
    return redirect("index")



def Register(request):
    if request.method == 'POST':  

        form_type = request.POST.get('form_type', None)

        if form_type == 'Register': # buttons name is checking
            first_name = request.POST.get('first_name', None)
            last_name = request.POST.get('last_name', None)
            email = request.POST.get('email', None)
            password1 = request.POST.get('password1', None)
            password2 = request.POST.get('password2', None)
            title = request.POST.get('title', None)
            unit = request.POST.get('unit', None)
            username = first_name + "_" + last_name  # Kullanıcı adını oluştur

            data = {
                'username': username,
                'first_name': first_name,
                'last_name': last_name,
                'email': email,
                'password1': password1,
                'password2': password2,
                'title': title,
                'unit': unit

            }
            print(data)
            form = SignupForm(data)  


            if form.is_valid():
                user = form.save(commit=True)  
                # user.is_active = False  
                # user.username = username 
                user.save()
                return redirect("index")
            else:
                messages.error(request, message=form.errors) 
                return redirect("index")
                
        else:
            messages.error(request, message = 'form type is not as expected')
            return redirect("index")
        
    return redirect("index")

def Profile(request):

    if request.user.is_authenticated:

        current_user = request.user
        user_profile = current_user.userprofile
        context = {
            'user_profile': user_profile,
        }
        
        return render(request, 'profile.html', context)
    else:
        return render(request, 'login_register.html')
    
def Logout(request):
    if request.user.is_authenticated:
        logout(request)
        return redirect("Login")
    else:
        return render(request, 'login_register.html')
    
def UpdateProfile(request):
    if request.user.is_authenticated:
        if request.method == 'POST':  
            form_type = request.POST.get('form_type', None)

            if form_type == 'UpdateProfile': # buttons name is checking
                email = request.POST.get('email', None)
                old_password = request.POST.get('old_password', None)
                new_password1 = request.POST.get('new_password1', None)
                new_password2 = request.POST.get('new_password2', None)

                # Check if passwords match
                if new_password1 != new_password2:
                    messages.error(request, "New passwords don't match.")
                    return redirect('Profile')  # Adjust the URL name for your view

                # Check if old password is correct
                if not request.user.check_password(old_password):
                    messages.error(request, "Old password is incorrect.")
                    return redirect('Profile')  # Adjust the URL name for your view

                # Set the new password
                request.user.set_password(new_password1)
                request.user.save()

                # Keep the user logged in after password change
                update_session_auth_hash(request, request.user)

                messages.success(request, 'Your password was successfully updated!')
                return redirect('Profile')  # Adjust the URL name for your view
    else:
        return redirect('Login')  # Redirect to login page if user is not authenticated

    # Add rendering logic here if needed

#  if user is not None:
#         login(request,user)
#         return Response({'ok':'True'},status=status.HTTP_200_OK)
#     else:
#         return Response({'ok':'False'},status=status.HTTP_401_UNAUTHORIZED)


