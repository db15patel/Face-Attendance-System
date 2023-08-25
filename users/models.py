from django.db import models
from django.contrib.auth.models import User

import datetime
# Create your models here.


class Present(models.Model):
	user=models.ForeignKey(User,on_delete=models.CASCADE)
	date = models.DateField(default=datetime.date.today)
	present=models.BooleanField(default=False)
	
class Time(models.Model):
	user=models.ForeignKey(User,on_delete=models.CASCADE)
	date = models.DateField(default=datetime.date.today)
	time=models.DateTimeField(null=True,blank=True)
	out=models.BooleanField(default=False)
	image=models.BinaryField(blank=True)


class User_details(models.Model):
	user=models.ForeignKey(User,on_delete=models.CASCADE)
	#user=models.CharField(max_length=100)
	email=models.CharField(max_length=120)
	contact=models.CharField(max_length=10)
	department=models.CharField(max_length=100)

class Holiday(models.Model):
	name=models.CharField(max_length=30)
	date=models.DateField()

	

