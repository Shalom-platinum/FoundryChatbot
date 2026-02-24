from django.contrib import admin
from .models import Conversation, Message, FileUpload, UserSettings


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ['title', 'user', 'model_used', 'created_at', 'updated_at']
    list_filter = ['model_used', 'created_at']
    search_fields = ['title', 'user__username']
    ordering = ['-updated_at']


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ['short_content', 'role', 'conversation', 'created_at']
    list_filter = ['role', 'created_at']
    search_fields = ['content']
    ordering = ['-created_at']
    
    def short_content(self, obj):
        return obj.content[:50] + '...' if len(obj.content) > 50 else obj.content
    short_content.short_description = 'Content'


@admin.register(FileUpload)
class FileUploadAdmin(admin.ModelAdmin):
    list_display = ['filename', 'user', 'file_type', 'file_size', 'created_at']
    list_filter = ['file_type', 'created_at']
    search_fields = ['filename']
    ordering = ['-created_at']


@admin.register(UserSettings)
class UserSettingsAdmin(admin.ModelAdmin):
    list_display = ['user', 'default_model', 'enable_web_search', 'enable_code_execution']
    list_filter = ['default_model', 'enable_web_search', 'enable_code_execution']
    search_fields = ['user__username']

