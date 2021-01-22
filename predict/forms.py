from django import forms
from predict.models import Predictions
# from django.contrib.auth.models import User

class Predict_Form(forms.ModelForm):
    class Meta:
        model = Predictions
        fields = ('xray_img',)

