from django.db import models

# Create your models here.
class UserComment(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    phone_number = models.CharField(max_length=15)
    comment = models.TextField()

class Image(models.Model):
    image = models.ImageField(upload_to='images/')

