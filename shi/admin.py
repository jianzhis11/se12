from django.contrib import admin
from shi.models.models import image
from shi.models.models import ChannelAttention, SpatialAttention, HybridSN
from shi.models.player.player import Player

# Register your models here.
admin.site.register(image)
admin.site.register(ChannelAttention)
admin.site.register(SpatialAttention)
admin.site.register(HybridSN)
admin.site.register(Player)

