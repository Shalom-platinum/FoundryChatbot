import logging
import os
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from rest_framework import status, viewsets, permissions
from rest_framework.decorators import api_view, permission_classes, action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken

from .models import Conversation, Message, FileUpload, UserSettings
from .serializers import (
    UserSerializer, UserSettingsSerializer,
    ConversationListSerializer, ConversationDetailSerializer,
    MessageSerializer, FileUploadSerializer,
    ChatRequestSerializer, SummarizeRequestSerializer
)
from .services import get_foundry_service, agent_orchestrator, AIProviderManager, get_ai_provider

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

def get_user_ai_provider(user):
    """
    Get the AI provider for a specific user based on their settings.
    Falls back to environment configuration if user settings are not set.
    """
    try:
        settings = UserSettings.objects.get(user=user)
        provider_name = settings.ai_provider
        
        # If using Azure OpenAI and user has custom credentials
        if provider_name == 'azure_openai' and settings.azure_openai_endpoint and settings.azure_openai_api_key:
            from .services.ai_provider import AzureOpenAIProvider
            return AzureOpenAIProvider(
                endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                deployment_name=settings.azure_openai_deployment or None
            )
        
        # Use the standard provider manager
        return get_ai_provider(provider_name)
    except UserSettings.DoesNotExist:
        # Fallback to default provider
        return get_ai_provider()
    except Exception as e:
        logger.error(f"Error getting AI provider for user: {e}")
        return get_ai_provider()


# =============================================================================
# Template Views (Frontend)
# =============================================================================

def login_view(request):
    """Login page view"""
    if request.user.is_authenticated:
        return redirect('chat_home')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('chat_home')
        else:
            return render(request, 'chat/login.html', {'error': 'Invalid credentials'})
    
    return render(request, 'chat/login.html')


def register_view(request):
    """Registration page view"""
    if request.user.is_authenticated:
        return redirect('chat_home')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        password_confirm = request.POST.get('password_confirm')
        
        if password != password_confirm:
            return render(request, 'chat/register.html', {'error': 'Passwords do not match'})
        
        if User.objects.filter(username=username).exists():
            return render(request, 'chat/register.html', {'error': 'Username already exists'})
        
        user = User.objects.create_user(username=username, email=email, password=password)
        UserSettings.objects.create(user=user)
        login(request, user)
        return redirect('chat_home')
    
    return render(request, 'chat/register.html')


def logout_view(request):
    """Logout view"""
    logout(request)
    return redirect('login')


@login_required
def chat_home(request, conversation_id=None):
    """Main chat interface"""
    conversations = Conversation.objects.filter(user=request.user)
    current_conversation = None
    messages = []
    
    if conversation_id:
        current_conversation = get_object_or_404(
            Conversation, id=conversation_id, user=request.user
        )
        messages = current_conversation.messages.all()
    
    # Get available models from user's selected provider
    try:
        ai_provider = get_user_ai_provider(request.user)
        models = ai_provider.get_available_models()
        available_providers = AIProviderManager.get_available_providers()
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        models = []
        available_providers = []
    
    # Get user settings
    settings, _ = UserSettings.objects.get_or_create(user=request.user)
    
    context = {
        'conversations': conversations,
        'current_conversation': current_conversation,
        'messages': messages,
        'models': models,
        'settings': settings,
        'available_providers': available_providers,
    }
    return render(request, 'chat/chat.html', context)


@login_required
def settings_view(request):
    """User settings page"""
    settings, _ = UserSettings.objects.get_or_create(user=request.user)
    
    if request.method == 'POST':
        settings.default_model = request.POST.get('default_model', settings.default_model)
        settings.system_prompt = request.POST.get('system_prompt', settings.system_prompt)
        settings.enable_web_search = request.POST.get('enable_web_search') == 'on'
        settings.enable_code_execution = request.POST.get('enable_code_execution') == 'on'
        
        # AI Provider settings
        settings.ai_provider = request.POST.get('ai_provider', settings.ai_provider)
        settings.azure_openai_endpoint = request.POST.get('azure_openai_endpoint', '').strip()
        settings.azure_openai_api_key = request.POST.get('azure_openai_api_key', '').strip()
        settings.azure_openai_deployment = request.POST.get('azure_openai_deployment', '').strip()
        
        settings.save()
        return redirect('settings')
    
    # Get available providers
    available_providers = AIProviderManager.get_available_providers()
    
    try:
        ai_provider = get_user_ai_provider(request.user)
        models = ai_provider.get_available_models()
    except Exception:
        models = []
    
    return render(request, 'chat/settings.html', {
        'settings': settings,
        'models': models,
        'available_providers': available_providers,
    })


# =============================================================================
# API Views
# =============================================================================

class RegisterAPIView(APIView):
    """API endpoint for user registration"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            refresh = RefreshToken.for_user(user)
            return Response({
                'user': serializer.data,
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ConversationViewSet(viewsets.ModelViewSet):
    """ViewSet for Conversation CRUD operations"""
    permission_classes = [IsAuthenticated]
    
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return ConversationDetailSerializer
        return ConversationListSerializer
    
    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
    
    @action(detail=True, methods=['delete'])
    def clear_messages(self, request, pk=None):
        """Clear all messages in a conversation"""
        conversation = self.get_object()
        conversation.messages.all().delete()
        return Response({'status': 'messages cleared'})


class MessageViewSet(viewsets.ModelViewSet):
    """ViewSet for Message operations"""
    serializer_class = MessageSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return Message.objects.filter(
            conversation__user=self.request.user
        )


class UserSettingsView(APIView):
    """API endpoint for user settings"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        settings, _ = UserSettings.objects.get_or_create(user=request.user)
        serializer = UserSettingsSerializer(settings)
        return Response(serializer.data)
    
    def put(self, request):
        settings, _ = UserSettings.objects.get_or_create(user=request.user)
        serializer = UserSettingsSerializer(settings, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_models(request):
    """Get available AI models for the user's selected provider"""
    try:
        ai_provider = get_user_ai_provider(request.user)
        models = ai_provider.get_available_models()
        return Response({
            'models': models,
            'provider': ai_provider.provider_name
        })
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_providers(request):
    """Get available AI providers"""
    try:
        providers = AIProviderManager.get_available_providers()
        settings, _ = UserSettings.objects.get_or_create(user=request.user)
        return Response({
            'providers': providers,
            'current_provider': settings.ai_provider
        })
    except Exception as e:
        logger.error(f"Error getting providers: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def switch_provider(request):
    """Switch AI provider for the current user"""
    provider_name = request.data.get('provider')
    
    if provider_name not in ['foundry_local', 'azure_openai']:
        return Response(
            {'error': 'Invalid provider'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        settings, _ = UserSettings.objects.get_or_create(user=request.user)
        settings.ai_provider = provider_name
        settings.save()
        
        # Get models for the new provider
        ai_provider = get_user_ai_provider(request.user)
        models = ai_provider.get_available_models()
        
        return Response({
            'success': True,
            'provider': provider_name,
            'models': models
        })
    except Exception as e:
        logger.error(f"Error switching provider: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def chat(request):
    """
    Main chat endpoint
    
    Processes a user message, optionally uses agentic tools,
    and returns AI response
    """
    serializer = ChatRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    data = serializer.validated_data
    message_content = data['message']
    conversation_id = data.get('conversation_id')
    model = data.get('model', 'phi-4-mini')
    use_web_search = data.get('use_web_search', False)
    use_code_execution = data.get('use_code_execution', False)
    
    try:
        # Get or create conversation
        if conversation_id:
            conversation = get_object_or_404(
                Conversation, id=conversation_id, user=request.user
            )
        else:
            # Create new conversation with title from first message
            title = message_content[:50] + '...' if len(message_content) > 50 else message_content
            conversation = Conversation.objects.create(
                user=request.user,
                title=title,
                model_used=model
            )
        
        # Save user message
        user_message = Message.objects.create(
            conversation=conversation,
            role='user',
            content=message_content
        )
        
        # Process with agentic tools if enabled
        tool_context = ''
        tool_calls = None
        tool_results = None
        
        if use_web_search or use_code_execution:
            agent_result = agent_orchestrator.process_with_tools(
                message_content,
                enable_web_search=use_web_search,
                enable_code_execution=use_code_execution
            )
            tool_context = agent_result.get('context', '')
            if agent_result['tool_calls']:
                tool_calls = agent_result['tool_calls']
                tool_results = agent_result['tool_results']
        
        # Build messages for AI
        messages = []
        
        # Get user settings for system prompt
        settings, _ = UserSettings.objects.get_or_create(user=request.user)
        messages.append({
            'role': 'system',
            'content': settings.system_prompt
        })
        
        # Add conversation history
        for msg in conversation.messages.all():
            messages.append({
                'role': msg.role,
                'content': msg.content
            })
        
        # Add tool context if available
        if tool_context:
            messages.append({
                'role': 'system',
                'content': f"The following information was gathered using tools:\n\n{tool_context}"
            })
        
        # Get AI response using user's selected provider
        ai_provider = get_user_ai_provider(request.user)
        ai_response = ai_provider.chat_completion(messages, model=model)
        
        # Save assistant message
        assistant_message = Message.objects.create(
            conversation=conversation,
            role='assistant',
            content=ai_response,
            tool_calls=tool_calls,
            tool_results=tool_results
        )
        
        # Update conversation
        conversation.model_used = model
        conversation.save()
        
        return Response({
            'conversation_id': str(conversation.id),
            'user_message': MessageSerializer(user_message).data,
            'assistant_message': MessageSerializer(assistant_message).data,
            'tool_calls': tool_calls,
            'tool_results': tool_results
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def summarize(request):
    """
    Summarization endpoint
    
    Summarizes text or uploaded file content
    """
    serializer = SummarizeRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    data = serializer.validated_data
    text = data.get('text', '')
    file_id = data.get('file_id')
    model = data.get('model', 'phi-4-mini')
    
    try:
        # If file_id provided, get text from file
        if file_id:
            file_upload = get_object_or_404(
                FileUpload, id=file_id, user=request.user
            )
            text = file_upload.extracted_text or ''
            if not text:
                return Response(
                    {'error': 'No text content found in file'},
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        ai_provider = get_user_ai_provider(request.user)
        summary = ai_provider.summarize_text(text, model=model)
        
        return Response({
            'summary': summary,
            'model': model,
            'original_length': len(text),
            'summary_length': len(summary)
        })
        
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_file(request):
    """
    File upload endpoint
    
    Handles file upload and text extraction
    """
    if 'file' not in request.FILES:
        return Response(
            {'error': 'No file provided'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    uploaded_file = request.FILES['file']
    conversation_id = request.data.get('conversation_id')
    
    try:
        # Basic text extraction (expand for more file types)
        content = ''
        file_type = uploaded_file.content_type
        
        if file_type in ['text/plain', 'text/markdown', 'text/csv']:
            content = uploaded_file.read().decode('utf-8')
        else:
            # For now, try to read as text
            try:
                content = uploaded_file.read().decode('utf-8')
            except UnicodeDecodeError:
                return Response(
                    {'error': 'Unable to extract text from file'},
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        conversation = None
        if conversation_id:
            conversation = get_object_or_404(
                Conversation, id=conversation_id, user=request.user
            )
        
        file_upload = FileUpload.objects.create(
            user=request.user,
            conversation=conversation,
            file=uploaded_file,
            filename=uploaded_file.name,
            file_type=file_type,
            file_size=uploaded_file.size,
            extracted_text=content
        )
        
        return Response(FileUploadSerializer(file_upload).data, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        logger.error(f"File upload error: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# =============================================================================
# AJAX Views for Template Frontend
# =============================================================================

@login_required
@require_http_methods(["POST"])
def ajax_send_message(request):
    """AJAX endpoint for sending messages from the template frontend"""
    import json
    
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    
    message_content = data.get('message', '')
    conversation_id = data.get('conversation_id')
    model = data.get('model', 'phi-4-mini')
    use_web_search = data.get('use_web_search', False)
    use_code_execution = data.get('use_code_execution', False)
    
    if not message_content:
        return JsonResponse({'error': 'Message is required'}, status=400)
    
    try:
        # Get or create conversation
        if conversation_id:
            conversation = get_object_or_404(
                Conversation, id=conversation_id, user=request.user
            )
        else:
            title = message_content[:50] + '...' if len(message_content) > 50 else message_content
            conversation = Conversation.objects.create(
                user=request.user,
                title=title,
                model_used=model
            )
        
        # Save user message
        user_message = Message.objects.create(
            conversation=conversation,
            role='user',
            content=message_content
        )
        
        # Process with agentic tools
        tool_context = ''
        tool_calls = None
        tool_results = None
        
        if use_web_search or use_code_execution:
            print("using tools")
            agent_result = agent_orchestrator.process_with_tools(
                message_content,
                enable_web_search=use_web_search,
                enable_code_execution=use_code_execution
            )
            tool_context = agent_result.get('context', '')
            if agent_result['tool_calls']:
                tool_calls = agent_result['tool_calls']
                tool_results = agent_result['tool_results']
        
        # Build messages
        messages = []
        settings, _ = UserSettings.objects.get_or_create(user=request.user)
        messages.append({'role': 'system', 'content': settings.system_prompt})
        
        for msg in conversation.messages.all():
            messages.append({'role': msg.role, 'content': msg.content})
        
        if tool_context:
            messages.append({
                'role': 'system',
                'content': f"Tool results:\n\n{tool_context}"
            })
        
        # Get AI response using user's selected provider
        ai_provider = get_user_ai_provider(request.user)
        ai_response = ai_provider.chat_completion(messages, model=model)
        
        # Save assistant message
        assistant_message = Message.objects.create(
            conversation=conversation,
            role='assistant',
            content=ai_response,
            tool_calls=tool_calls,
            tool_results=tool_results
        )
        
        conversation.model_used = model
        conversation.save()
        
        return JsonResponse({
            'success': True,
            'conversation_id': str(conversation.id),
            'user_message': {
                'id': str(user_message.id),
                'content': user_message.content,
                'created_at': user_message.created_at.isoformat()
            },
            'assistant_message': {
                'id': str(assistant_message.id),
                'content': assistant_message.content,
                'created_at': assistant_message.created_at.isoformat()
            },
            'tool_calls': tool_calls,
            'tool_results': tool_results
        })
        
    except Exception as e:
        logger.error(f"AJAX chat error: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@login_required
@require_http_methods(["POST"])
def ajax_new_conversation(request):
    """Create a new conversation"""
    conversation = Conversation.objects.create(
        user=request.user,
        title='New Chat'
    )
    return JsonResponse({
        'success': True,
        'conversation_id': str(conversation.id),
        'title': conversation.title
    })


@login_required
@require_http_methods(["DELETE"])
def ajax_delete_conversation(request, conversation_id):
    """Delete a conversation"""
    conversation = get_object_or_404(
        Conversation, id=conversation_id, user=request.user
    )
    conversation.delete()
    return JsonResponse({'success': True})


@login_required
@require_http_methods(["PUT"])
def ajax_rename_conversation(request, conversation_id):
    """Rename a conversation"""
    import json
    
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    
    title = data.get('title', '')
    if not title:
        return JsonResponse({'error': 'Title is required'}, status=400)
    
    conversation = get_object_or_404(
        Conversation, id=conversation_id, user=request.user
    )
    conversation.title = title
    conversation.save()
    
    return JsonResponse({'success': True, 'title': title})


@login_required
@require_http_methods(["POST"])
def ajax_upload_file(request):
    """AJAX endpoint for file upload from template frontend"""
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file provided'}, status=400)
    
    uploaded_file = request.FILES['file']
    conversation_id = request.POST.get('conversation_id')
    
    try:
        # Basic text extraction
        content = ''
        file_type = uploaded_file.content_type or 'text/plain'
        
        try:
            content = uploaded_file.read().decode('utf-8')
        except UnicodeDecodeError:
            return JsonResponse(
                {'error': 'Unable to extract text from file. Please upload a text file.'},
                status=400
            )
        
        conversation = None
        if conversation_id:
            conversation = get_object_or_404(
                Conversation, id=conversation_id, user=request.user
            )
        
        file_upload = FileUpload.objects.create(
            user=request.user,
            conversation=conversation,
            file=uploaded_file,
            filename=uploaded_file.name,
            file_type=file_type,
            file_size=uploaded_file.size,
            extracted_text=content
        )
        
        return JsonResponse({
            'success': True,
            'id': str(file_upload.id),
            'filename': file_upload.filename,
            'file_type': file_upload.file_type,
            'file_size': file_upload.file_size,
            'extracted_text': content
        })
        
    except Exception as e:
        logger.error(f"AJAX file upload error: {e}")
        return JsonResponse({'error': str(e)}, status=500)

