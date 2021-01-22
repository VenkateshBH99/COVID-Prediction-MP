from django.shortcuts import render

# Create your views here.
import csv,io
from django.shortcuts import render
from .forms import Predict_Form
from predict.data_provider import *
from accounts.models import UserProfileInfo
from django.shortcuts import get_object_or_404, redirect, render
from django.http import HttpResponseRedirect, HttpResponse
from django.contrib.auth.decorators import login_required,permission_required
from django.urls import reverse
from django.contrib import messages
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing import image
import numpy as np
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
from django.core.files.storage import default_storage

# from keras.models import load_weights
# from attn_layer import AttentionLayer
# import tensorflow_hub as hub
from pathlib import Path
from django.core.files.storage import FileSystemStorage


@login_required(login_url='/')
def PredictRisk(request,pk):
    predicted = False
    predictions={}
    if request.session.has_key('user_id'):
        u_id = request.session['user_id']

    if request.method == 'POST':
        form = Predict_Form(data=request.POST)
        profile = get_object_or_404(UserProfileInfo, pk=pk)

        print("hello--------")
        print(form.is_valid())
        if form.is_valid():
            print("hiiiii--------")
            img = form.save(commit=False)

            print("img:::",img)
            # if 'xray_img' in request.FILES:
            #     img.xray_img = request.FILES['xray_img']
            myfile = request.FILES['xray_img']
            fs = FileSystemStorage()
            print("myfile:",myfile)
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)
            print("Up:",uploaded_file_url)
            # print('img_1:',img.xray_img)
            # fs = FileSystemStorage()
            # filename = fs.save(img.xray_img.name, img.xray_img)
            # file_name_2 = default_storage.save(str(img.xray_img), img.xray_img)
            # file_url = default_storage.url(file_name_2)
            # BASE_DIR = Path(__file__).resolve().parent.parent
            # tmp=str(os.path.join(BASE_DIR, "accounts/media"))
            img = load_img(uploaded_file_url, target_size=(224, 224))
            image = img_to_array(img)
            image = image.reshape((1,image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)  # (1,299,299,3)
            # print(image.shape)

            val=[image]
            val=np.array(val)
            val = val .astype('float32')
            val=np.rollaxis(val,1,0)
            val=val[0]
            print(np.shape(val))
            print (val)
            dir = os.getcwd()+'/predict/'
            fil=os.path.join(dir, 'modelLayer.h5')
            print(fil)
            model1 = load_model(fil)
            # reloaded_model = tf.keras.experimental.load_from_saved_model('path_to_my_model.h5', custom_objects={'KerasLayer':hub.KerasLayer})

            print("hellooooooo")
            y_pred=model1.predict(val)
            y_pred1= np.argmax(y_pred, axis=1)
            print("prediction:",y_pred1)
            y_pred1=y_pred1[0]
            # features = [[ form.cleaned_data['age'], form.cleaned_data['sex'], form.cleaned_data['cp'], form.cleaned_data['resting_bp'], form.cleaned_data['serum_cholesterol'],
            # form.cleaned_data['fasting_blood_sugar'], form.cleaned_data['resting_ecg'], form.cleaned_data['max_heart_rate'], form.cleaned_data['exercise_induced_angina'],
            # form.cleaned_data['st_depression'], form.cleaned_data['st_slope'], form.cleaned_data['number_of_vessels'], form.cleaned_data['thallium_scan_results']]]

            # print("Before------",features)
            # standard_scalar = GetStandardScalarForHeart()
            # features = standard_scalar.transform(features)
            # print("Hell0-------",features)
            # SVCClassifier,LogisticRegressionClassifier,NaiveBayesClassifier,DecisionTreeClassifier,NeuralNetworkClassifier,KNNClassifier=GetAllClassifiersForHeart()


            # predictions = {'SVC': str(SVCClassifier.predict(features)[0]),
            # 'LogisticRegression': str(LogisticRegressionClassifier.predict(features)[0]),
            #  'NaiveBayes': str(NaiveBayesClassifier.predict(features)[0]),
            #  'DecisionTree': str(DecisionTreeClassifier.predict(features)[0]),
            #   'NeuralNetwork': str(NeuralNetworkClassifier.predict(features)[0]),
            #   'KNN': str(KNNClassifier.predict(features)[0]),
            #   }
            pred = form.save(commit=False)
            print("Pred:",pred)
            # l=[predictions['SVC'],predictions['LogisticRegression'],predictions['NaiveBayes'],predictions['DecisionTree'],predictions['NeuralNetwork'],predictions['KNN']]
            # count=l.count('1')
            #
            result=False
            if y_pred1==1:
                result=True
                pred.num=1
            else:
                pred.num=0
            #
            # if count>=3:
            #     result=True
            #     pred.num=1
            # else:
            #     pred.num=0
            #
            # pred.profile = profile
            #

            # pred.save()
            predicted = True
            #
            colors={}
            #
            if y_pred1==1:
                colors['covid']="table-danger"
            else:
                colors['covid']="table-success"
            #
            # if predictions['LogisticRegression']=='0':
            #     colors['LR']="table-success"
            # else:
            #     colors['LR']="table-danger"
            #
            # if predictions['NaiveBayes']=='0':
            #     colors['NB']="table-success"
            # else:
            #     colors['NB']="table-danger"
            #
            # if predictions['DecisionTree']=='0':
            #     colors['DT']="table-success"
            # else:
            #     colors['DT']="table-danger"
            #
            # if predictions['NeuralNetwork']=='0':
            #     colors['NN']="table-success"
            # else:
            #     colors['NN']="table-danger"
            #
            # if predictions['KNN']=='0':
            #     colors['KNN']="table-success"
            # else:
            #     colors['KNN']="table-danger"

    if predicted:
        # pass
        return render(request, 'predict.html',
                      {'form': form,'predicted': predicted,'user_id':u_id,'predictions':str(y_pred1),'result':result,'colors':colors})

    else:
        form = Predict_Form()

        return render(request, 'predict.html',
                      {'form': form,'predicted': predicted,'user_id':u_id,'predictions':predictions})
