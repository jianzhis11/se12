from django.shortcuts import render
from shi.models.models import image
from shi.views.utils.utils import getNewName
import shutil

def index(request):
    return render(request, "multiends/web.html")

def g(request):
    pic=''
    # 获取上传的图片对象
    if request.method == 'POST':
        pic=request.FILES.get('img')
    new_name = getNewName('img')
    # pic.chunks() 返回一个生成器，存储该文件的内容
    load = '%s/image/%s' % ('/home/lxy/se12/shi/media', new_name)
    new_path = '%s/image/%s' % ('/home/lxy/se12/shi/static', new_name)
    with open(load, 'wb') as f:
    # 这里返回一个生成器需要通过遍历才可以得到内容
        for content in pic.chunks():
                f.write(content)
    shutil.copy(load, new_path)
    # 在数据库中添加该上传记录
    image.objects.create(img='/image/%s' % new_name)
    content={
            'img': '/static/image/'+new_name
            }
    return render(request, 'multiends/web.html', content)
