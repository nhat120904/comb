# logs/models.py

from django.db import models
from django.contrib.auth.models import User

class Staff(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    department = models.CharField(max_length=100)

    def __str__(self):
        return self.user.get_full_name()

class ErrorLog(models.Model):
    staff = models.ForeignKey(Staff, on_delete=models.CASCADE, related_name='error_logs')
    error_type = models.CharField(max_length=100)
    description = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.error_type} by {self.staff} at {self.timestamp}"
