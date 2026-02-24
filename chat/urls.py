from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from . import views

# REST API router
router = DefaultRouter()
router.register(r'conversations', views.ConversationViewSet, basename='conversation')
router.register(r'messages', views.MessageViewSet, basename='message')

urlpatterns = [
    # Template views (Frontend)
    path('', views.chat_home, name='chat_home'),
    path('chat/<uuid:conversation_id>/', views.chat_home, name='chat_detail'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('settings/', views.settings_view, name='settings'),
    
    # AJAX endpoints for template frontend
    path('ajax/send-message/', views.ajax_send_message, name='ajax_send_message'),
    path('ajax/new-conversation/', views.ajax_new_conversation, name='ajax_new_conversation'),
    path('ajax/conversation/<uuid:conversation_id>/delete/', views.ajax_delete_conversation, name='ajax_delete_conversation'),
    path('ajax/conversation/<uuid:conversation_id>/rename/', views.ajax_rename_conversation, name='ajax_rename_conversation'),
    path('ajax/upload/', views.ajax_upload_file, name='upload_file'),
    
    # REST API endpoints
    path('api/', include(router.urls)),
    path('api/auth/register/', views.RegisterAPIView.as_view(), name='api_register'),
    path('api/auth/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/auth/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/settings/', views.UserSettingsView.as_view(), name='api_settings'),
    path('api/models/', views.get_models, name='api_models'),
    path('api/providers/', views.get_providers, name='api_providers'),
    path('api/providers/switch/', views.switch_provider, name='api_switch_provider'),
    path('api/chat/', views.chat, name='api_chat'),
    path('api/summarize/', views.summarize, name='api_summarize'),
    path('api/upload/', views.upload_file, name='api_upload'),
]
