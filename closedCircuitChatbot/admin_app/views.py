# views.py

from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .forms import UploadFileForm
import os
from .dataEmbedding import * 
def AdminMain(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            new_form = UploadFileForm()
            form_type = request.POST.get('form_type', None)
            if form_type == 'load_data_set':
                path_ = os.path.join(os.getcwd(),"admin_app/admin_data")
                print(path_)
                form = UploadFileForm(request.POST, request.FILES)
                if form.is_valid():
                    dataset_file = form.cleaned_data['dataset_file']
                    fs = FileSystemStorage(location= os.path.join(os.getcwd(),"admin_app/admin_data"))
                    filename = fs.save(dataset_file.name, dataset_file)
                    uploaded_file_url = fs.url(filename)
                    directory(uploaded_file_url)
                    return render(request, "admin.html", {'form': new_form})
                else:
                    return render(request, "admin.html", {'form': new_form})
            else: 
                return render(request, "admin.html", {'form': new_form})
        else:
            form = UploadFileForm()
        return render(request, "admin.html", {'form': form})
    print("bitti")
    return redirect('login')  # Redirect to login if not authenticated
