# logs/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('staff/', views.staff_list, name='staff_list'),
    path('staff/<int:staff_id>/', views.staff_detail, name='staff_detail'),
    path('error/<int:error_id>/', views.error_detail, name='error_detail'),
    path('reports/', views.reports, name='reports'),
    path('accounts/logout/', views.logout_view, name='logout'),  # {{ edit_1 }}
    path('accounts/login/', views.manager_login, name='manager_login'),  # {{ edit_2 }}
    path('accounts/register/', views.manager_register, name='manager_register'),  # {{ edit_3 }}
]
