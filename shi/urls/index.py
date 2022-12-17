from django.urls import path, include
from shi.views.index import index, g, func


urlpatterns = [
    path("", index, name="index"),
    path("g", g, name="g"),
    path("func", func, name="func"),
]

