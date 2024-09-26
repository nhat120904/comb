# logs/admin.py

from django.contrib import admin
from .models import Staff, ErrorLog

admin.site.register(Staff)
admin.site.register(ErrorLog)