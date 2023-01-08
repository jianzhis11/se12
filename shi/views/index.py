from django.shortcuts import render
from django.http import JsonResponse
from django.contrib import messages
from django.contrib.auth.models import User
from shi.models.player.player import Player
from django.shortcuts import redirect
from django.contrib.auth import authenticate, login, logout
from shi.models.models import image, ChannelAttention, SpatialAttention, HybridSN
from shi.views.utils.utils import getNewName
import shutil
import spectral as hsi
import torch.optim as optim
import torch
from sklearn.decomposition import PCA
import scipy.io as sio
import numpy as np
from PIL import Image

def index(request):
    return render(request, "multiends/login.html")

def toregister(request):
    return render(request, "multiends/register.html")

def register(request):
    data = request.GET
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    password_confirm = data.get("password_confirm", "").strip()
    if password != password_confirm:
        messages.success(request, "两个密码不一致")
        return redirect('toregister')
    if User.objects.filter(username=username).exists():
        messages.success(request, "用户名已存在")
        return redirect('toregister')
    user = User(username=username)
    user.set_password(password)
    user.save()
    Player.objects.create(user=user, photo="https://s3.bmp.ovh/imgs/2023/01/07/b1ff4a481caea4d9.jpg")
    login(request, user)
    return render(request, "multiends/web.html")

def signin(request):
    data = request.GET
    username = data.get('username')
    password = data.get('password')
    user = authenticate(username=username, password=password)
    if not user:
        return redirect('index')
    login(request, user)
    return render(request, "multiends/web.html")

def signout(request):
    user = request.user
    if not user.is_authenticated:
        messages.success(request, "密码错误")
        return redirect('index')
    logout(request)
    return redirect('index')


def func(request):
    return render(request, "multiends/func.html")

def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX



def g(request):
    # 获取上传的 .mat 文件
    if request.method == "POST":
        myFile = request.FILES.get("file")
    new_name = getNewName('mat')
    load = '%s/%s' % ('/home/lxy/se12/shi/dataset', new_name)
    with open(load, 'wb') as f:
        for content in myFile.chunks():
            f.write(content)

    # load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=HybridSN()
    state_dict=torch.load('/home/lxy/se12/shi/trained_models/model_salinasA.pt', map_location={'cuda:0':'cpu'})
    model.load_state_dict(state_dict)
    model=model.to(device)
    # load the original image
    X = sio.loadmat(load)['salinasA_corrected']
    y = sio.loadmat('/home/lxy/se12/shi/dataset/SalinasA_gt.mat')['salinasA_gt']

    height = y.shape[0]
    width = y.shape[1]

    pca_components=30
    patch_size = 25
    X = applyPCA(X, numComponents= pca_components)
    X = padWithZeros(X, patch_size//2)

    # 逐像素预测类别
    outputs = np.zeros((height,width))
    for i in range(height):
        print(i)
        for j in range(width):
            if int(y[i,j]) == 0:
                continue
            else:
                image_patch = X[i:i+patch_size, j:j+patch_size, :]
                image_patch = image_patch.reshape(1,image_patch.shape[0],image_patch.shape[1], image_patch.shape[2], 1)
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
                prediction = model(X_test_image)
                prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                outputs[i][j] = prediction+1


    new_name = getNewName('img')
    print(new_name)
    new_path = '%s/image/%s' % ('/home/lxy/se12/shi/static', new_name)
    load_path = '%s/image/%s' % ('/static', new_name)

    hsi.save_rgb(new_path, outputs.astype(int), colors=hsi.spy_colors)
    content={
            'img': load_path
            }
    return render(request, 'multiends/func.html', content)


