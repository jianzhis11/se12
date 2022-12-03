from django.shortcuts import render
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
    return render(request, "multiends/web.html")

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
    # pic=''
    # # 获取上传的图片对象
    # if request.method == 'POST':
    #     pic=request.FILES.get('img')
    # new_name = getNewName('img')
    # # pic.chunks() 返回一个生成器，存储该文件的内容
    # load = '%s/image/%s' % ('/home/lxy/se12/shi/media', new_name)
    # new_path = '%s/image/%s' % ('/home/lxy/se12/shi/static', new_name)
    # with open(load, 'wb') as f:
    #     # 这里返回一个生成器需要通过遍历才可以得到内容
    #     for content in pic.chunks():
    #         f.write(content)
    # shutil.copy(load, new_path)
    # # 在数据库中添加该上传记录
    # image.objects.create(img='/image/%s' % new_name)

    # load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=HybridSN()
    state_dict=torch.load('/home/lxy/se12/shi/trained_models/model_salinasA.pt', map_location={'cuda:0':'cpu'})
    model.load_state_dict(state_dict)
    model=model.to(device)
    # load the original image
    X = sio.loadmat('/home/lxy/se12/shi/dataset/SalinasA_corrected.mat')['salinasA_corrected']
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
    return render(request, 'multiends/web.html', content)


