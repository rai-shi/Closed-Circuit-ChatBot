
from django import forms

class UploadFileForm(forms.Form):
    dataset_file = forms.FileField()