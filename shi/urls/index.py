from django.urls import path, include
from shi.views.index import index, g, func,signin,register,signout,toregister


urlpatterns = [
    path("", index, name="index"),
    path("login",signin,name="signin"),
    path("register",register,name="register"),
    path("toregister",toregister,name="toregister"),
    path("logout",signout,name="signout"),
    path("g", g, name="g"),
    path("func", func, name="func"),
]

