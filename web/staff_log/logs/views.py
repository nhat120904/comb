# logs/views.py

from django.shortcuts import render, get_object_or_404, redirect
from .models import Staff, ErrorLog
from django.contrib.auth.decorators import login_required
from django.db.models import Count
from django.utils import timezone
from datetime import timedelta
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages
from django.contrib.auth.models import User
from .forms import ManagerRegisterForm

@login_required
def dashboard(request):
    recent_errors = ErrorLog.objects.order_by('-timestamp')[:10]
    error_counts = ErrorLog.objects.values('error_type').annotate(count=Count('error_type'))
    total_errors = ErrorLog.objects.count()

    context = {
        'recent_errors': recent_errors,
        'error_counts': error_counts,
        'total_errors': total_errors,
    }
    return render(request, 'logs/dashboard.html', context)

# logs/views.py

@login_required
def staff_list(request):
    staff_members = Staff.objects.annotate(error_count=Count('error_logs'))
    context = {
        'staff_members': staff_members,
    }
    return render(request, 'logs/staff_list.html', context)

# logs/views.py

@login_required
def staff_detail(request, staff_id):
    staff = get_object_or_404(Staff, id=staff_id)
    error_logs = staff.error_logs.order_by('-timestamp')
    context = {
        'staff': staff,
        'error_logs': error_logs,
    }
    return render(request, 'logs/staff_detail.html', context)

# logs/views.py

@login_required
def error_detail(request, error_id):
    error = get_object_or_404(ErrorLog, id=error_id)
    context = {
        'error': error,
    }
    return render(request, 'logs/error_detail.html', context)

# logs/views.py

from django.http import HttpResponse
import csv

@login_required
def reports(request):
    if request.method == 'GET':
        # Fetch all error logs
        errors = ErrorLog.objects.all()

        # Create the HttpResponse object with CSV headers.
        response = HttpResponse(
            content_type='text/csv',
            headers={'Content-Disposition': 'attachment; filename="error_reports.csv"'},
        )

        writer = csv.writer(response)
        writer.writerow(['Staff Name', 'Error Type', 'Description', 'Timestamp'])

        for error in errors:
            writer.writerow([
                error.staff.user.get_full_name(),
                error.error_type,
                error.description,
                error.timestamp
            ])

        return response

    return render(request, 'logs/reports.html')

def manager_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome back, {user.username}!")
                return redirect('dashboard')
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid form submission.")
    else:
        form = AuthenticationForm()
    return render(request, 'logs/manager_login.html', {'form': form})



def manager_register(request):
    if request.method == 'POST':
        form = ManagerRegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f"Account created successfully for {user.username}")
            return redirect('dashboard')
        else:
            messages.error(request, "Registration failed. Please check the form.")
    else:
        form = ManagerRegisterForm()
    return render(request, 'logs/manager_register.html', {'form': form})

# logs/views.py
from django.views.decorators.http import require_POST

@require_POST
def logout_view(request):
    logout(request)
    messages.success(request, "You have been logged out.")
    return redirect('manager_login')