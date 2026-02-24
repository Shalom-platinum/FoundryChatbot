from django.db import models
from django.contrib.auth.models import User
import uuid


class Conversation(models.Model):
    """Model to store chat conversations"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversations')
    title = models.CharField(max_length=255, blank=True, default='New Chat')
    model_used = models.CharField(max_length=100, default='phi-4-mini')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return f"{self.title} - {self.user.username}"

    def get_message_count(self):
        return self.messages.count()


class Message(models.Model):
    """Model to store individual messages in a conversation"""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(
        Conversation, on_delete=models.CASCADE, related_name='messages'
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    # For storing tool/agent results
    tool_calls = models.JSONField(null=True, blank=True)
    tool_results = models.JSONField(null=True, blank=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."


class FileUpload(models.Model):
    """Model to store uploaded files for summarization"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploads')
    conversation = models.ForeignKey(
        Conversation, on_delete=models.CASCADE, 
        related_name='files', null=True, blank=True
    )
    file = models.FileField(upload_to='uploads/')
    filename = models.CharField(max_length=255)
    file_type = models.CharField(max_length=50)
    file_size = models.IntegerField()
    extracted_text = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.filename


class UserSettings(models.Model):
    """User-specific settings"""
    
    AI_PROVIDER_CHOICES = [
        ('foundry_local', 'Foundry Local'),
        ('azure_openai', 'Azure OpenAI'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='settings')
    default_model = models.CharField(max_length=100, default='phi-4-mini')
    system_prompt = models.TextField(
        default="You are a helpful AI assistant powered by Microsoft Foundry Local."
    )
    enable_web_search = models.BooleanField(default=False)
    enable_code_execution = models.BooleanField(default=False)
    
    # AI Provider settings
    ai_provider = models.CharField(
        max_length=20,
        choices=AI_PROVIDER_CHOICES,
        default='foundry_local',
        help_text="Select the AI provider to use"
    )
    
    # Azure OpenAI settings (optional - can use environment variables instead)
    azure_openai_endpoint = models.CharField(
        max_length=500,
        blank=True,
        default='',
        help_text="Azure OpenAI endpoint URL (optional - uses env var if empty)"
    )
    azure_openai_api_key = models.CharField(
        max_length=500,
        blank=True,
        default='',
        help_text="Azure OpenAI API key (optional - uses env var if empty)"
    )
    azure_openai_deployment = models.CharField(
        max_length=100,
        blank=True,
        default='',
        help_text="Azure OpenAI deployment name (optional - uses env var if empty)"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Settings for {self.user.username}"
    
    def get_ai_provider_display_name(self):
        """Return the display name for the current AI provider"""
        return dict(self.AI_PROVIDER_CHOICES).get(self.ai_provider, 'Unknown')

