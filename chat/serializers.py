from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Conversation, Message, FileUpload, UserSettings


class UserSerializer(serializers.ModelSerializer):
    """Serializer for User model"""
    password = serializers.CharField(write_only=True, min_length=8)

    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'password', 'first_name', 'last_name']
        read_only_fields = ['id']

    def create(self, validated_data):
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data.get('email', ''),
            password=validated_data['password'],
            first_name=validated_data.get('first_name', ''),
            last_name=validated_data.get('last_name', '')
        )
        # Create default user settings
        UserSettings.objects.create(user=user)
        return user


class UserSettingsSerializer(serializers.ModelSerializer):
    """Serializer for UserSettings model"""
    azure_openai_api_key = serializers.CharField(write_only=True, required=False, allow_blank=True)
    
    class Meta:
        model = UserSettings
        fields = [
            'default_model', 'system_prompt', 
            'enable_web_search', 'enable_code_execution',
            'ai_provider', 'azure_openai_endpoint', 
            'azure_openai_api_key', 'azure_openai_deployment',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']


class MessageSerializer(serializers.ModelSerializer):
    """Serializer for Message model"""
    class Meta:
        model = Message
        fields = [
            'id', 'conversation', 'role', 'content', 
            'tool_calls', 'tool_results', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class ConversationListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for listing conversations"""
    message_count = serializers.SerializerMethodField()
    last_message = serializers.SerializerMethodField()

    class Meta:
        model = Conversation
        fields = [
            'id', 'title', 'model_used', 
            'message_count', 'last_message',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_message_count(self, obj):
        return obj.messages.count()

    def get_last_message(self, obj):
        last_msg = obj.messages.last()
        if last_msg:
            return {
                'role': last_msg.role,
                'content': last_msg.content[:100] + '...' if len(last_msg.content) > 100 else last_msg.content,
                'created_at': last_msg.created_at
            }
        return None


class ConversationDetailSerializer(serializers.ModelSerializer):
    """Detailed serializer for single conversation with messages"""
    messages = MessageSerializer(many=True, read_only=True)
    
    class Meta:
        model = Conversation
        fields = [
            'id', 'title', 'model_used', 
            'messages', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class FileUploadSerializer(serializers.ModelSerializer):
    """Serializer for FileUpload model"""
    class Meta:
        model = FileUpload
        fields = [
            'id', 'filename', 'file_type', 'file_size',
            'extracted_text', 'created_at'
        ]
        read_only_fields = ['id', 'filename', 'file_type', 'file_size', 'created_at']


class ChatRequestSerializer(serializers.Serializer):
    """Serializer for chat request"""
    message = serializers.CharField(required=True)
    conversation_id = serializers.UUIDField(required=False, allow_null=True)
    model = serializers.CharField(required=False, default='phi-4-mini')
    use_web_search = serializers.BooleanField(required=False, default=False)
    use_code_execution = serializers.BooleanField(required=False, default=False)


class SummarizeRequestSerializer(serializers.Serializer):
    """Serializer for summarization request"""
    text = serializers.CharField(required=False, allow_blank=True)
    file_id = serializers.UUIDField(required=False, allow_null=True)
    model = serializers.CharField(required=False, default='phi-4-mini')

    def validate(self, data):
        if not data.get('text') and not data.get('file_id'):
            raise serializers.ValidationError(
                "Either 'text' or 'file_id' must be provided"
            )
        return data
