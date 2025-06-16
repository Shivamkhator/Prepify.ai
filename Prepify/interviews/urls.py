from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('interview/', views.interview_session, name='interview'),
    path('report/<int:interview_id>/', views.view_report, name='view_report'),
    
    # API endpoints
    path('api/start-interview/', views.start_interview, name='start_interview'),
    path('api/process-audio/', views.process_audio, name='process_audio'),
    path('api/end-interview/<int:interview_id>/', views.end_interview, name='end_interview'),
]