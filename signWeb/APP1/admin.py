from django.contrib import admin
from django.contrib.admin import AdminSite
from .import models
from django.db import models

# Register your models here.
from .models import UserComment
admin.site.register(UserComment)

from .models import Image
admin.site.register(Image)
