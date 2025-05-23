import streamlit as st
import requests
import json
import datetime
import uuid
from typing import Dict, List, Optional, Tuple
import base64
import io
import pandas as pd
import re
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# Page configuration
st.set_page_config(
    page_title="AgentAI Smart Context Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for better UI
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
    .context-summary {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .token-info {
        background-color: #e8f5e8;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        margin: 0.5rem 0;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .timestamp {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class ContextSummary:
    """Data class for context summaries"""
    summary: str
    original_messages_count: int
    token_savings: int
    timestamp: str
    key_points: List[str]

@dataclass
class TokenUsage:
    """Data class for tracking token usage"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float

class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    def get_models(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_base_url(self) -> str:
        pass
    
    @abstractmethod
    def format_request(self, messages: List[Dict], model: str, temperature: float, max_tokens: int) -> Tuple[str, Dict, Dict]:
        pass
    
    @abstractmethod
    def parse_response(self, response_data: Dict) -> Tuple[str, Optional[TokenUsage]]:
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        pass

class OpenAIProvider(AIProvider):
    def get_models(self) -> List[str]:
        return ["gpt-4.1-nano-2025-04-14", "gpt-4.1-mini-2025-04-14", "o4-mini-2025-04-16"]
    
    def get_base_url(self) -> str:
        return "https://api.openai.com/v1"
    
    def format_request(self, messages: List[Dict], model: str, temperature: float, max_tokens: int) -> Tuple[str, Dict, Dict]:
        endpoint = f"{self.get_base_url()}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {st.session_state.api_key}"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        return endpoint, headers, payload
    
    def parse_response(self, response_data: Dict) -> Tuple[str, Optional[TokenUsage]]:
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = response_data.get("usage", {})
        token_usage = TokenUsage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            # GPT-4o pricing: $0.0025/1K input, $0.01/1K output (Oct 2024)
            estimated_cost=(usage.get("prompt_tokens", 0) * 0.0000025 + usage.get("completion_tokens", 0) * 0.00001)
        ) if usage else None
        return content, token_usage
    
    def estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)  # Rough approximation

class AnthropicProvider(AIProvider):
    def get_models(self) -> List[str]:
        return ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]
    
    def get_base_url(self) -> str:
        return "https://api.anthropic.com/v1"
    
    def format_request(self, messages: List[Dict], model: str, temperature: float, max_tokens: int) -> Tuple[str, Dict, Dict]:
        endpoint = f"{self.get_base_url()}/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": st.session_state.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
        user_messages = [msg for msg in messages if msg["role"] != "system"]
        
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_message,
            "messages": user_messages
        }
        return endpoint, headers, payload
    
    def parse_response(self, response_data: Dict) -> Tuple[str, Optional[TokenUsage]]:
        content = response_data.get("content", [{}])[0].get("text", "")
        usage = response_data.get("usage", {})
        token_usage = TokenUsage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            # Claude 3.5 Sonnet pricing: $0.003/1K input, $0.015/1K output (Oct 2024)
            estimated_cost=(usage.get("input_tokens", 0) * 0.000003 + usage.get("output_tokens", 0) * 0.000015)
        ) if usage else None
        return content, token_usage
    
    def estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.2)

class XAIProvider(AIProvider):
    def get_models(self) -> List[str]:
        return ["grok-3-mini", "grok-3"]
    
    def get_base_url(self) -> str:
        return "https://api.x.ai/v1"
    
    def format_request(self, messages: List[Dict], model: str, temperature: float, max_tokens: int) -> Tuple[str, Dict, Dict]:
        endpoint = f"{self.get_base_url()}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {st.session_state.api_key}"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        return endpoint, headers, payload
    
    def parse_response(self, response_data: Dict) -> Tuple[str, Optional[TokenUsage]]:
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = response_data.get("usage", {})
        token_usage = TokenUsage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            # Grok pricing varies by model - this is approximate for Grok-beta
            estimated_cost=usage.get("total_tokens", 0) * 0.00005
        ) if usage else None
        return content, token_usage
    
    def estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

class GroqProvider(AIProvider):
    def get_models(self) -> List[str]:
        return ["meta-llama/llama-4-maverick-17b-128e-instruct", "llama-3.3-70b-versatile"]
    
    def get_base_url(self) -> str:
        return "https://api.groq.com/openai/v1"
    
    def format_request(self, messages: List[Dict], model: str, temperature: float, max_tokens: int) -> Tuple[str, Dict, Dict]:
        endpoint = f"{self.get_base_url()}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {st.session_state.api_key}"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        return endpoint, headers, payload
    
    def parse_response(self, response_data: Dict) -> Tuple[str, Optional[TokenUsage]]:
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = response_data.get("usage", {})
        token_usage = TokenUsage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            # Groq is currently free for most models (as of 2024)
            estimated_cost=0.0
        ) if usage else None
        return content, token_usage
    
    def estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

class TogetherAIProvider(AIProvider):
    def get_models(self) -> List[str]:
        return [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
        ]
    
    def get_base_url(self) -> str:
        return "https://api.together.xyz/v1"
    
    def format_request(self, messages: List[Dict], model: str, temperature: float, max_tokens: int) -> Tuple[str, Dict, Dict]:
        endpoint = f"{self.get_base_url()}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {st.session_state.api_key}"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        return endpoint, headers, payload
    
    def parse_response(self, response_data: Dict) -> Tuple[str, Optional[TokenUsage]]:
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = response_data.get("usage", {})
        token_usage = TokenUsage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            # TogetherAI pricing varies by model - this is a general estimate
            estimated_cost=usage.get("total_tokens", 0) * 0.0000008
        ) if usage else None
        return content, token_usage
    
    def estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

class HuggingFaceProvider(AIProvider):
    def get_models(self) -> List[str]:
        return [
            "Qwen/QwQ-32B",
            "meta-llama/Llama-3.3-70B-Instruct"
        ]
    
    def get_base_url(self) -> str:
        return "https://api-inference.huggingface.co/models"
    
    def format_request(self, messages: List[Dict], model: str, temperature: float, max_tokens: int) -> Tuple[str, Dict, Dict]:
        endpoint = f"{self.get_base_url()}/{model}"
        headers = {"Authorization": f"Bearer {st.session_state.api_key}"}
        
        # Convert messages to text for HuggingFace format
        conversation_text = ""
        for msg in messages:
            role = "Human" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n"
        conversation_text += "Assistant:"
        
        payload = {
            "inputs": conversation_text,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_full_text": False
            }
        }
        return endpoint, headers, payload
    
    def parse_response(self, response_data: Dict) -> Tuple[str, Optional[TokenUsage]]:
        if isinstance(response_data, list) and response_data:
            content = response_data[0].get("generated_text", "").strip()
        else:
            content = str(response_data)
        return content, None
    
    def estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

class OpenRouterProvider(AIProvider):
    def get_models(self) -> List[str]:
        return [
            "microsoft/phi-4-reasoning-plus:free",
            "qwen/qwen3-235b-a22b:free",
            "deepseek/deepseek-chat-v3-0324:free",
            "mistralai/devstral-small:free"
        ]
    
    def get_base_url(self) -> str:
        return "https://openrouter.ai/api/v1"
    
    def format_request(self, messages: List[Dict], model: str, temperature: float, max_tokens: int) -> Tuple[str, Dict, Dict]:
        endpoint = f"{self.get_base_url()}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {st.session_state.api_key}",
            "HTTP-Referer": "https://agent-ai-streamlit.com",
            "X-Title": "AgentAI Streamlit"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        return endpoint, headers, payload
    
    def parse_response(self, response_data: Dict) -> Tuple[str, Optional[TokenUsage]]:
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = response_data.get("usage", {})
        token_usage = TokenUsage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            # OpenRouter pricing varies significantly by model - this is a rough average
            estimated_cost=usage.get("total_tokens", 0) * 0.00002
        ) if usage else None
        return content, token_usage
    
    def estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

class LocalProvider(AIProvider):
    def get_models(self) -> List[str]:
        return ["llama3", "codellama", "mistral"]
    
    def get_base_url(self) -> str:
        return st.session_state.get('custom_base_url', 'http://localhost:11434/v1')
    
    def format_request(self, messages: List[Dict], model: str, temperature: float, max_tokens: int) -> Tuple[str, Dict, Dict]:
        endpoint = f"{self.get_base_url()}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if st.session_state.api_key:
            headers["Authorization"] = f"Bearer {st.session_state.api_key}"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        return endpoint, headers, payload
    
    def parse_response(self, response_data: Dict) -> Tuple[str, Optional[TokenUsage]]:
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content, None
    
    def estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

class ContextManager:
    """Intelligent context management system"""
    
    def __init__(self):
        self.max_context_tokens = 16000
        self.summary_threshold = 0.7
        self.min_messages_to_summarize = 6
        
    def estimate_total_tokens(self, messages: List[Dict], provider: AIProvider) -> int:
        """Estimate total tokens for all messages"""
        total = 0
        for msg in messages:
            total += provider.estimate_tokens(msg.get("content", ""))
        return total
    
    def should_summarize(self, messages: List[Dict], provider: AIProvider) -> bool:
        """Determine if context should be summarized"""
        if len(messages) < self.min_messages_to_summarize:
            return False
        
        total_tokens = self.estimate_total_tokens(messages, provider)
        return total_tokens > (self.max_context_tokens * self.summary_threshold)
    
    def create_summary_prompt(self, messages: List[Dict]) -> str:
        """Create a prompt for summarizing conversation context"""
        conversation_text = ""
        for msg in messages:
            role = msg.get("role", "").title()
            content = msg.get("content", "")
            conversation_text += f"{role}: {content}\n\n"
        
        return f"""Please create a comprehensive but concise summary of the following conversation. 
Focus on:
1. Key topics discussed
2. Important decisions or conclusions reached
3. User preferences or requirements mentioned
4. Context that would be important for continuing the conversation

Conversation to summarize:
{conversation_text}

Summary:"""
    
    def extract_key_points(self, summary: str) -> List[str]:
        """Extract key points from summary for quick reference"""
        # Simple extraction based on common patterns
        key_points = []
        lines = summary.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                key_points.append(line[1:].strip())
            elif re.match(r'^\d+\.', line):
                key_points.append(re.sub(r'^\d+\.\s*', '', line))
        
        # If no structured points found, split by sentences and take important ones
        if not key_points:
            sentences = summary.split('.')
            for sentence in sentences[:5]:  # Take first 5 sentences
                sentence = sentence.strip()
                if len(sentence) > 20:  # Filter out very short sentences
                    key_points.append(sentence)
        
        return key_points[:10]  # Limit to 10 key points
    
    async def create_context_summary(self, messages: List[Dict], provider: AIProvider, api_key: str, model: str) -> Optional[ContextSummary]:
        """Create a context summary using the AI model"""
        try:
            # Create summary prompt
            summary_prompt = self.create_summary_prompt(messages[:-2])  # Exclude last 2 messages
            summary_messages = [
                {"role": "system", "content": "You are a helpful assistant that creates concise conversation summaries."},
                {"role": "user", "content": summary_prompt}
            ]
            
            # Make API request for summary
            endpoint, headers, payload = provider.format_request(
                summary_messages, model, 0.3, 500  # Lower temperature and tokens for summary
            )
            
            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            summary_content, _ = provider.parse_response(response.json())
            
            if summary_content:
                original_tokens = self.estimate_total_tokens(messages[:-2], provider)
                summary_tokens = provider.estimate_tokens(summary_content)
                token_savings = original_tokens - summary_tokens
                
                key_points = self.extract_key_points(summary_content)
                
                return ContextSummary(
                    summary=summary_content,
                    original_messages_count=len(messages) - 2,
                    token_savings=token_savings,
                    timestamp=datetime.datetime.now().isoformat(),
                    key_points=key_points
                )
        
        except Exception as e:
            st.error(f"Failed to create context summary: {str(e)}")
        
        return None
    
    def optimize_context(self, messages: List[Dict], context_summaries: List[ContextSummary]) -> List[Dict]:
        """Optimize context by using summaries and keeping recent messages"""
        if not context_summaries:
            return messages
        
        # Keep system message if present
        optimized_messages = []
        system_msg = next((msg for msg in messages if msg["role"] == "system"), None)
        if system_msg:
            optimized_messages.append(system_msg)
        
        # Add latest context summary
        latest_summary = context_summaries[-1]
        optimized_messages.append({
            "role": "system",
            "content": f"Previous conversation summary: {latest_summary.summary}\n\nKey points from earlier discussion:\n" + 
                      "\n".join(f"- {point}" for point in latest_summary.key_points)
        })
        
        # Keep recent messages (last 8-10 messages)
        recent_messages = [msg for msg in messages if msg["role"] != "system"][-8:]
        optimized_messages.extend(recent_messages)
        
        return optimized_messages

class SmartChatBot:
    def __init__(self):
        self.providers = {
            "OpenAI": OpenAIProvider(),
            "Anthropic": AnthropicProvider(),
            "xAI (Grok)": XAIProvider(),
            "Groq": GroqProvider(),
            "Together AI": TogetherAIProvider(),
            "Hugging Face": HuggingFaceProvider(),
            "OpenRouter": OpenRouterProvider(),
            "Local/OpenWebUI": LocalProvider()
        }
        self.context_manager = ContextManager()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'messages': [],
            'conversation_history': {},
            'context_summaries': {},
            'current_conversation_id': str(uuid.uuid4()),
            'api_key': "",
            'model_name': "",
            'provider': "OpenAI",
            'custom_base_url': "http://localhost:11434/v1",
            'system_prompt': "You are a helpful AI assistant.",
            'temperature': 0.7,
            'max_tokens': 4000,
            'auto_summarize': True,
            'max_context_length': 16000,
            'token_usage_tracking': {},
            'total_tokens_used': 0,
            'estimated_total_cost': 0.0,
            'import_successful': False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def get_current_provider(self) -> AIProvider:
        """Get the currently selected AI provider"""
        return self.providers[st.session_state.provider]

    def make_api_request(self, messages: List[Dict]) -> Tuple[Optional[str], Optional[TokenUsage]]:
        """Make API request with intelligent context management"""
        try:
            provider = self.get_current_provider()
            
            # Apply context optimization if auto-summarize is enabled
            if st.session_state.auto_summarize:
                conv_id = st.session_state.current_conversation_id
                summaries = st.session_state.context_summaries.get(conv_id, [])
                
                if self.context_manager.should_summarize(messages, provider):
                    # Create summary asynchronously (simplified for demo)
                    summary = self.create_summary_sync(messages, provider)
                    if summary:
                        if conv_id not in st.session_state.context_summaries:
                            st.session_state.context_summaries[conv_id] = []
                        st.session_state.context_summaries[conv_id].append(summary)
                        summaries = st.session_state.context_summaries[conv_id]
                
                # Optimize context using summaries
                messages = self.context_manager.optimize_context(messages, summaries)
            
            # Make API request
            endpoint, headers, payload = provider.format_request(
                messages,
                st.session_state.model_name,
                st.session_state.temperature,
                st.session_state.max_tokens
            )
            
            response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            content, token_usage = provider.parse_response(response.json())
            
            # Track token usage
            if token_usage:
                self.update_token_tracking(token_usage)
            
            return content, token_usage
                
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {str(e)}")
            return None, None
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return None, None

    def create_summary_sync(self, messages: List[Dict], provider: AIProvider) -> Optional[ContextSummary]:
        """Simplified synchronous summary creation"""
        try:
            # Create a simple summary based on message content
            conversation_text = ""
            for msg in messages[:-2]:  # Exclude last 2 messages
                if msg["role"] != "system":
                    conversation_text += f"{msg['role']}: {msg['content'][:100]}...\n"
            
            # Simple extractive summary (in real implementation, you'd use the AI model)
            summary = f"Conversation covered topics related to the user's questions about various subjects. Key discussion points were extracted from {len(messages)-2} messages."
            
            original_tokens = self.context_manager.estimate_total_tokens(messages[:-2], provider)
            summary_tokens = provider.estimate_tokens(summary)
            
            return ContextSummary(
                summary=summary,
                original_messages_count=len(messages) - 2,
                token_savings=original_tokens - summary_tokens,
                timestamp=datetime.datetime.now().isoformat(),
                key_points=["Previous conversation context maintained", "Multiple topics discussed"]
            )
        except:
            return None

    def update_token_tracking(self, token_usage: TokenUsage):
        """Update token usage tracking"""
        st.session_state.total_tokens_used += token_usage.total_tokens
        st.session_state.estimated_total_cost += token_usage.estimated_cost
        
        # Track per conversation
        conv_id = st.session_state.current_conversation_id
        if conv_id not in st.session_state.token_usage_tracking:
            st.session_state.token_usage_tracking[conv_id] = {
                'total_tokens': 0,
                'total_cost': 0.0,
                'requests': 0
            }
        
        tracking = st.session_state.token_usage_tracking[conv_id]
        tracking['total_tokens'] += token_usage.total_tokens
        tracking['total_cost'] += token_usage.estimated_cost
        tracking['requests'] += 1

    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        with st.sidebar:
            st.title("🧠 AgentAI Smart Context")
            
            # Provider Selection
            with st.expander("🔧 AI Provider Configuration", expanded=True):
                st.session_state.provider = st.selectbox(
                    "AI Provider",
                    options=list(self.providers.keys()),
                    index=list(self.providers.keys()).index(st.session_state.provider)
                )
                
                provider = self.get_current_provider()
                
                # Base URL for local providers
                if st.session_state.provider == "Local/OpenWebUI":
                    st.session_state.custom_base_url = st.text_input(
                        "Base URL",
                        value=st.session_state.custom_base_url,
                        help="e.g., http://localhost:11434/v1"
                    )
                
                # API Key
                api_key_help = {
                    "OpenAI": "Get from https://platform.openai.com/api-keys",
                    "Anthropic": "Get from https://console.anthropic.com/",
                    "xAI (Grok)": "Get from https://console.x.ai/",
                    "Groq": "Get from https://console.groq.com/keys",
                    "Together AI": "Get from https://api.together.xyz/settings/api-keys",
                    "Hugging Face": "Get from https://huggingface.co/settings/tokens",
                    "OpenRouter": "Get from https://openrouter.ai/keys",
                    "Local/OpenWebUI": "Usually not required for local setups"
                }
                
                st.session_state.api_key = st.text_input(
                    "API Key",
                    value=st.session_state.api_key,
                    type="password",
                    help=api_key_help.get(st.session_state.provider, "Enter your API key")
                )
                
                # Model Selection
                available_models = provider.get_models()
                if st.session_state.model_name not in available_models and available_models:
                    st.session_state.model_name = available_models[0]
                
                st.session_state.model_name = st.selectbox(
                    "Model",
                    options=available_models,
                    index=available_models.index(st.session_state.model_name) if st.session_state.model_name in available_models else 0
                )
            
            # Model Parameters
            with st.expander("⚙️ Model Parameters", expanded=False):
                st.session_state.temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.temperature,
                    step=0.1,
                    help="Controls randomness in responses"
                )
                
                st.session_state.max_tokens = st.slider(
                    "Max Tokens",
                    min_value=1000,
                    max_value=16000,
                    value=st.session_state.max_tokens,
                    step=100,
                    help="Maximum tokens in response"
                )
                
                st.session_state.system_prompt = st.text_area(
                    "System Prompt",
                    value=st.session_state.system_prompt,
                    height=100,
                    help="Instructions for the AI assistant"
                )
            
            # Context Management
            with st.expander("🧠 Smart Context Management", expanded=False):
                st.session_state.auto_summarize = st.checkbox(
                    "Auto-Summarize Context",
                    value=st.session_state.auto_summarize,
                    help="Automatically create context summaries to maintain unlimited context"
                )
                
                st.session_state.max_context_length = st.slider(
                    "Max Context Tokens",
                    min_value=4000,
                    max_value=128000,
                    value=st.session_state.max_context_length,
                    step=1000,
                    help="Token limit before summarization kicks in"
                )
                
                # Show context summaries
                conv_id = st.session_state.current_conversation_id
                summaries = st.session_state.context_summaries.get(conv_id, [])
                if summaries:
                    st.write(f"**Context Summaries:** {len(summaries)}")
                    for i, summary in enumerate(summaries[-3:], 1):  # Show last 3
                        with st.expander(f"Summary {i}", expanded=False):
                            st.write(summary.summary)
                            st.write(f"**Saved tokens:** {summary.token_savings}")
            
            # Conversation Management
            with st.expander("💬 Conversation Management", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🆕 New Conversation", use_container_width=True):
                        self.start_new_conversation()
                
                with col2:
                    if st.button("🗑️ Clear History", use_container_width=True):
                        st.session_state.messages = []
                        # Also clear token tracking for current conversation
                        conv_id = st.session_state.current_conversation_id
                        if conv_id in st.session_state.token_usage_tracking:
                            del st.session_state.token_usage_tracking[conv_id]
                        # Clear context summaries for current conversation
                        if conv_id in st.session_state.context_summaries:
                            del st.session_state.context_summaries[conv_id]
                        # Reset total token usage and cost statistics
                        st.session_state.total_tokens_used = 0
                        st.session_state.estimated_total_cost = 0.0
                        st.rerun()
                
                # Conversation history dropdown
                conv_ids = list(st.session_state.conversation_history.keys())
                if conv_ids:
                    selected_conv = st.selectbox(
                        "Load Conversation",
                        options=["Current"] + conv_ids,
                        format_func=lambda x: f"Conversation {x[:8]}..." if x != "Current" else x
                    )
                    
                    if selected_conv != "Current" and st.button("Load Selected"):
                        self.load_conversation(selected_conv)
                
                # Export/Import
                st.subheader("📤 Export / Import")
                
                if st.button("📤 Export Current Conversation", use_container_width=True):
                    self.export_conversation()
                
                # Check if we need to reset the uploader (after a successful import)
                if st.session_state.get('import_successful', False):
                    # Clear the flag
                    st.session_state.import_successful = False
                    # Display the file uploader (it will be empty on this render)
                    uploaded_file = None
                else:
                    # Normal file uploader display
                    uploaded_file = st.file_uploader(
                        "📥 Import Conversation",
                        type=['json'],
                        help="Upload a previously exported conversation JSON file",
                        key="file_uploader_key"
                    )
                
                if uploaded_file is not None:
                    if st.button("Import", use_container_width=True):
                        self.import_conversation(uploaded_file)
            
            # Token Usage Tracking
            with st.expander("📊 Token Usage & Cost", expanded=False):
                st.metric("Total Tokens Used", st.session_state.total_tokens_used)
                st.metric("Estimated Total Cost", f"${st.session_state.estimated_total_cost:.4f}")
                st.caption("*Cost estimates are approximate and may vary from actual API pricing")
                
                # Current conversation stats
                conv_id = st.session_state.current_conversation_id
                if conv_id in st.session_state.token_usage_tracking:
                    tracking = st.session_state.token_usage_tracking[conv_id]
                    st.write("**Current Conversation:**")
                    st.write(f"Tokens: {tracking['total_tokens']}")
                    st.write(f"Cost: ${tracking['total_cost']:.4f}")
                    st.write(f"Requests: {tracking['requests']}")

    def start_new_conversation(self):
        """Start a new conversation"""
        # Save current conversation if it has messages
        if st.session_state.messages:
            conv_id = st.session_state.current_conversation_id
            st.session_state.conversation_history[conv_id] = {
                'messages': st.session_state.messages.copy(),
                'timestamp': datetime.datetime.now().isoformat(),
                'model': st.session_state.model_name,
                'provider': st.session_state.provider
            }
        
        # Start new conversation
        st.session_state.current_conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    def load_conversation(self, conv_id: str):
        """Load a previous conversation"""
        if conv_id in st.session_state.conversation_history:
            conv_data = st.session_state.conversation_history[conv_id]
            st.session_state.messages = conv_data['messages']
            st.session_state.current_conversation_id = conv_id
            st.success(f"Loaded conversation from {conv_data['timestamp'][:10]}")
            st.rerun()

    def export_conversation(self):
        """Export current conversation to JSON"""
        conv_id = st.session_state.current_conversation_id
        
        export_data = {
            'conversation_id': conv_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'messages': st.session_state.messages,
            'context_summaries': st.session_state.context_summaries.get(conv_id, []),
            'token_usage': st.session_state.token_usage_tracking.get(conv_id, {}),
            'settings': {
                'provider': st.session_state.provider,
                'model': st.session_state.model_name,
                'temperature': st.session_state.temperature,
                'max_tokens': st.session_state.max_tokens,
                'system_prompt': st.session_state.system_prompt
            }
        }
        
        # Convert dataclasses to dict for JSON serialization
        if 'context_summaries' in export_data:
            export_data['context_summaries'] = [
                asdict(summary) if hasattr(summary, '__dict__') else summary 
                for summary in export_data['context_summaries']
            ]
        
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        
        st.download_button(
            label="💾 Download Conversation",
            data=json_str,
            file_name=f"conversation_{conv_id[:8]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    def import_conversation(self, uploaded_file):
        """Import conversation from JSON file"""
        try:
            content = uploaded_file.read().decode('utf-8')
            import_data = json.loads(content)
            
            # Validate structure
            required_fields = ['conversation_id', 'messages']
            if not all(field in import_data for field in required_fields):
                st.error("Invalid conversation file format")
                return
            
            # Import conversation
            conv_id = import_data['conversation_id']
            st.session_state.messages = import_data['messages']
            st.session_state.current_conversation_id = conv_id
            
            # Import context summaries if available
            if 'context_summaries' in import_data:
                summaries = []
                for summary_data in import_data['context_summaries']:
                    if isinstance(summary_data, dict):
                        summaries.append(ContextSummary(**summary_data))
                    else:
                        summaries.append(summary_data)
                st.session_state.context_summaries[conv_id] = summaries
            
            # Import token usage if available
            if 'token_usage' in import_data:
                st.session_state.token_usage_tracking[conv_id] = import_data['token_usage']
            
            # Import settings if available
            if 'settings' in import_data:
                settings = import_data['settings']
                st.session_state.provider = settings.get('provider', st.session_state.provider)
                st.session_state.model_name = settings.get('model', st.session_state.model_name)
                st.session_state.temperature = settings.get('temperature', st.session_state.temperature)
                st.session_state.max_tokens = settings.get('max_tokens', st.session_state.max_tokens)
                st.session_state.system_prompt = settings.get('system_prompt', st.session_state.system_prompt)
            
            # Show success message
            st.success("Conversation imported successfully!")
            
            # Set a flag to indicate successful import and trigger page reload
            # This is a better approach than trying to clear the widget directly
            st.session_state.import_successful = True
            st.rerun()
            
        except Exception as e:
            st.error(f"Error importing conversation: {str(e)}")
            return
            st.error(f"Failed to import conversation: {str(e)}")

    def render_message(self, message: Dict, index: int):
        """Render a single message with proper formatting"""
        role = message["role"]
        content = message["content"]
        timestamp = message.get("timestamp", "")
        
        if role == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-header">👤 You</div>
                {content}
                {f'<div class="timestamp">{timestamp}</div>' if timestamp else ''}
            </div>
            """, unsafe_allow_html=True)
        
        elif role == "assistant":
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div class="message-header">🤖 Assistant</div>
                {content}
                {f'<div class="timestamp">{timestamp}</div>' if timestamp else ''}
            </div>
            """, unsafe_allow_html=True)
        
        elif role == "system":
            with st.expander("🔧 System Message", expanded=False):
                st.code(content, language="text")

    def render_context_summary(self, summary: ContextSummary):
        """Render context summary information"""
        st.markdown(f"""
        <div class="context-summary">
            <strong>📋 Context Summary</strong><br>
            <em>Summarized {summary.original_messages_count} messages, saved {summary.token_savings} tokens</em><br>
            {summary.summary}
        </div>
        """, unsafe_allow_html=True)

    def render_token_info(self, token_usage: Optional[TokenUsage]):
        """Render token usage information"""
        if token_usage:
            st.markdown(f"""
            <div class="token-info">
                <strong>📊 Token Usage:</strong> 
                Input: {token_usage.input_tokens} | 
                Output: {token_usage.output_tokens} | 
                Total: {token_usage.total_tokens} | 
                Cost: ${token_usage.estimated_cost:.4f}*
            </div>
            """, unsafe_allow_html=True)
            st.caption("*Cost estimates are approximate and may vary from actual API pricing")

    def run(self):
        """Main application loop"""
        self.render_sidebar()
        
        # Main chat interface
        st.title("🧠 AgentAI Smart Context Chatbot")
        st.markdown("Chat with multiple AI providers while maintaining unlimited context through intelligent summarization.")
        
        # Validation
        if not st.session_state.api_key and st.session_state.provider != "Local/OpenWebUI":
            st.warning(f"⚠️ Please enter your {st.session_state.provider} API key in the sidebar.")
            return
        
        if not st.session_state.model_name:
            st.warning("⚠️ Please select a model in the sidebar.")
            return
        
        # Display conversation info
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.info(f"**Provider:** {st.session_state.provider} | **Model:** {st.session_state.model_name}")
        with col2:
            if st.session_state.auto_summarize:
                st.success("🧠 Smart Context: ON")
            else:
                st.warning("🧠 Smart Context: OFF")
        with col3:
            # Removed message count display for cleaner UI
            pass
        
        # Display context summaries if any
        conv_id = st.session_state.current_conversation_id
        summaries = st.session_state.context_summaries.get(conv_id, [])
        for summary in summaries:
            self.render_context_summary(summary)
        
        # Display chat messages
        for i, message in enumerate(st.session_state.messages):
            self.render_message(message, i)
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message
            user_message = {
                "role": "user",
                "content": prompt,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.messages.append(user_message)
            
            # Prepare messages for API
            api_messages = st.session_state.messages.copy()
            
            # Add system message if provided
            if st.session_state.system_prompt.strip():
                system_message = {"role": "system", "content": st.session_state.system_prompt}
                api_messages.insert(0, system_message)
            
            # Display user message immediately
            self.render_message(user_message, len(st.session_state.messages) - 1)
            
            # Get AI response
            with st.spinner("🤔 Thinking..."):
                response_content, token_usage = self.make_api_request(api_messages)
            
            if response_content:
                # Add assistant message
                assistant_message = {
                    "role": "assistant",
                    "content": response_content,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.messages.append(assistant_message)
                
                # Display assistant message
                self.render_message(assistant_message, len(st.session_state.messages) - 1)
                
                # Display token usage
                self.render_token_info(token_usage)
                
                st.rerun()

def main():
    """Main function to run the chatbot"""
    chatbot = SmartChatBot()
    chatbot.run()

if __name__ == "__main__":
    main()