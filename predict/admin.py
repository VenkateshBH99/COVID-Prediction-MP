from django.contrib import admin
from predict.models import Predictions
from django import forms

class Prediction(admin.ModelAdmin):
    list_display=('xray_img','num')
admin.site.register(Predictions,Prediction)
# Register your models here.
