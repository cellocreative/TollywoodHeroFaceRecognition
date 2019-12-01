from django.shortcuts import render
from . import custom
def home(request):
    return render(request,'index.html')
def result(request):
    file = request.FILES['hero']
    prediction = custom.get_hero_name(file)
    return render(request,'result.html',{'prediction':prediction})
