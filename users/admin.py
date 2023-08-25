from django.contrib import admin
from .models import Time,Present,User_details,Holiday

# Register your models here.
admin.site.register(Time)
admin.site.register(Present)
admin.site.register(User_details)
admin.site.register(Holiday)