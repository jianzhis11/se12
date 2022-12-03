# Generated by Django 3.2.8 on 2022-12-03 03:18

from django.db import migrations, models
import torch.nn.modules.module


class Migration(migrations.Migration):

    dependencies = [
        ('shi', '0003_alter_image_img'),
    ]

    operations = [
        migrations.CreateModel(
            name='ChannelAttention',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
            ],
            bases=(torch.nn.modules.module.Module, models.Model),
        ),
        migrations.CreateModel(
            name='HybridSN',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
            ],
            bases=(torch.nn.modules.module.Module, models.Model),
        ),
        migrations.CreateModel(
            name='SpatialAttention',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
            ],
            bases=(torch.nn.modules.module.Module, models.Model),
        ),
    ]