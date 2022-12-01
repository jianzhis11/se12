from django.urls import path, include
from shi.views.index import index, g


urlpatterns = [
    path("", index, name="index"),
    path("g", g, name="g"),
]

