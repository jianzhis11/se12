from django.db import models

class image(models.Model):
    '''从后台上传图片'''
    # upload_to 指定文件上传的地址
    img = models.FileField(upload_to='image/')

