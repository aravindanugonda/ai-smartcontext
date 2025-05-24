# 🧠 AgentAI Smart Context Chatbot

A sophisticated multi-provider AI chatbot with intelligent context management and unlimited conversation history through automatic summarization.

## ✨ Features

### 🎯 Multi-Provider Support
- **OpenAI** - gpt-4.1-nano-2025-04-14, gpt-4.1-mini-2025-04-14, o4-mini-2025-04-16
- **Anthropic** - claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022
- **xAI (Grok)** - grok-3-mini, grok-3
- **Groq** - meta-llama/llama-4-maverick-17b-128e-instruct, llama-3.3-70b-versatile
- **Together AI** - meta-llama/Llama-3.3-70B-Instruct-Turbo-Free, deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free
- **Hugging Face** - Qwen/QwQ-32B, meta-llama/Llama-3.3-70B-Instruct
- **OpenRouter** - microsoft/phi-4-reasoning-plus:free, qwen/qwen3-235b-a22b:free, deepseek/deepseek-chat-v3-0324:free, mistralai/devstral-small:free
- **Local/OpenWebUI** - Compatible with Ollama and other local model servers (llama3, codellama, mistral)

### 🧠 Smart Context Management
- **Automatic Summarization** - Intelligently summarizes conversations when they exceed 70% of token limit
- **Token Optimization** - Reduces token usage by 60-80% through intelligent compression
- **Context Preservation** - Key points and context are preserved across summarizations
- **Unlimited History** - Effectively unlimited conversation length through smart compression
- **Thinking Model Support** - Extracts clean output from reasoning models (o1, QwQ, etc.)
- **Fallback Mechanisms** - Creates extractive summaries when AI summarization fails
- **Real-time Monitoring** - Live token count display with threshold indicators
- **Silent Operation** - Context management works transparently in the background

### 💾 Conversation Management
- **Export/Import** - Save and load conversations in JSON format with full context data
- **Multiple Conversations** - Manage multiple conversation threads with unique IDs
- **History Tracking** - Browse and reload previous conversations with metadata
- **Context Summaries Export** - Includes summarization data and token savings in exports
- **Settings Preservation** - Exports include model settings and configuration
- **Backward Compatibility** - Handles legacy conversation formats gracefully

### ⚡ Advanced Configuration
- **Model Parameters** - Adjust temperature (0.0-2.0), max tokens (1000-16000), and system prompts
- **Provider Settings** - Configure API keys and custom endpoints for local models
- **Token Tracking** - Real-time per-conversation and total token usage monitoring
- **Smart Thresholds** - Configurable summarization triggers (4000-128000 tokens)
- **Message Formatting** - Beautiful UI with role-based styling and timestamps
- **Error Handling** - Robust error handling with fallback mechanisms
- **Session Persistence** - Maintains state across browser sessions

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd ai-smartcontext
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run chatbot.py
   ```

4. **Open your browser**
   - The application will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in your terminal

## 🔧 Configuration

### API Keys Setup

#### OpenAI
1. Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Create a new API key
3. Enter it in the sidebar under "AI Provider Configuration"

#### Anthropic (Claude)
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Generate an API key
3. Enter it in the sidebar

#### xAI (Grok)
1. Visit [xAI Console](https://console.x.ai/)
2. Get your API key
3. Enter it in the sidebar

#### Groq
1. Visit [Groq Console](https://console.groq.com/keys)
2. Create an API key
3. Enter it in the sidebar

#### Other Providers
- **Together AI**: [API Keys](https://api.together.xyz/settings/api-keys)
- **Hugging Face**: [Tokens](https://huggingface.co/settings/tokens)
- **OpenRouter**: [Keys](https://openrouter.ai/keys)

### Local Setup (Ollama/OpenWebUI)
1. Install [Ollama](https://ollama.ai/) or set up OpenWebUI
2. Configure the base URL in the sidebar (default: `http://localhost:11434/v1`)
3. API key is usually not required for local setups

## 📖 Usage Guide

### Basic Usage
1. **Select Provider**: Choose your AI provider in the sidebar
2. **Enter API Key**: Add your API key for the selected provider
3. **Choose Model**: Select from available models
4. **Start Chatting**: Type your message and press Enter

### Smart Context Features
- **Auto-Summarize**: Toggle automatic context summarization (enabled by default)
- **Token Limits**: Set maximum context tokens before summarization (4000-128000)
- **Real-time Status**: View current token usage vs threshold in sidebar
- **Active Summaries**: Monitor number of active summaries working silently
- **Threshold Visualization**: See percentage of context limit reached
- **Background Processing**: Summarization happens automatically without user intervention

### Conversation Management
- **New Conversation**: Start a fresh conversation thread
- **Export**: Download current conversation as JSON
- **Import**: Upload and restore previous conversations
- **Load History**: Switch between saved conversation threads

### Advanced Settings
- **Temperature**: Control response randomness (0.0 = deterministic, 2.0 = creative)
- **Max Tokens**: Set maximum response length
- **System Prompt**: Customize AI behavior and instructions

## 🏗️ Architecture

### Core Components

#### AIProvider Classes
- Abstract base class with provider-specific implementations
- Handles API formatting, response parsing, and token estimation
- Supports different authentication methods and request formats

#### ContextManager
- Intelligent conversation summarization
- Token estimation and optimization
- Context compression with key point extraction

#### SmartChatBot
- Main application controller
- Session state management
- UI rendering and user interaction

### Smart Context System
1. **Token Monitoring** - Continuously tracks conversation length with provider-specific estimation
2. **Summarization Trigger** - Activates when 70% of configurable token limit is reached
3. **AI-Powered Compression** - Uses the same AI model to create intelligent summaries
4. **Context Optimization** - Replaces old messages with summaries while keeping recent 8-10 messages
5. **Key Point Extraction** - Identifies and preserves critical conversation elements
6. **Fallback Strategy** - Creates extractive summaries when AI summarization fails
7. **Thinking Model Handling** - Filters out reasoning process from models like o1 and QwQ

## 📊 Token Usage & Smart Context Analytics

The application provides comprehensive tracking and optimization:

### Real-time Monitoring
- **Input Tokens** - Tokens sent to the API for each request
- **Output Tokens** - Tokens received from the API in responses
- **Total Usage** - Combined token consumption per conversation and globally
- **Context Size** - Current conversation token count vs configured limits
- **Threshold Indicators** - Visual progress toward summarization trigger

### Smart Context Benefits
- **Token Savings** - Achieves 60-80% reduction in context tokens through summarization
- **Cost Optimization** - Significant API cost reduction for long conversations
- **Unlimited Length** - No practical limit on conversation duration
- **Context Preservation** - Maintains conversation coherence through intelligent compression
- **Provider Agnostic** - Works consistently across all supported AI providers
- **Export Analytics** - Detailed summarization statistics in conversation exports

### Technical Implementation
- **Adaptive Estimation** - Provider-specific token counting algorithms
- **Intelligent Chunking** - Preserves recent messages while compressing history
- **Error Resilience** - Multiple fallback strategies ensure reliability
- **Performance Optimization** - Background processing doesn't interrupt user experience

## 🔒 Privacy & Security

- **Local Storage** - All conversations stored locally in browser session
- **API Key Security** - Keys are not logged or transmitted except to respective APIs
- **No Data Collection** - Application doesn't collect or store user data
- **Export Control** - You control all conversation exports and imports

## 🛠️ Troubleshooting

### Common Issues

#### API Connection Errors
- Verify API key is correct and active
- Check internet connection
- Ensure provider service is operational

#### Model Not Available
- Some models may have regional restrictions
- Try alternative models from the same provider
- Check provider documentation for model availability

#### Summarization Issues
- Disable auto-summarize if experiencing problems with specific models
- Manually start new conversations for fresh context if needed
- Check token limits match your provider's model context windows
- Verify API key has sufficient quota for summarization requests
- Review fallback summary creation if AI summarization consistently fails

### Performance Tips
- **Use Groq** for fastest inference with free tier models
- **Enable auto-summarization** for cost-effective long conversations
- **Choose appropriate models** - balance between capability and speed
- **Use local providers** for privacy and unlimited usage
- **Configure token limits** based on your provider's context windows
- **Monitor token usage** to optimize costs across different providers
- **Export conversations** regularly to preserve important discussions

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- Additional AI provider integrations
- Enhanced summarization algorithms
- UI/UX improvements
- Performance optimizations
- Documentation updates

## 📝 License

This project is open source. Please ensure you comply with the terms of service of the AI providers you use.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review provider documentation for API-specific issues
3. Ensure your API keys have sufficient credits/quota

## 🔮 Future Enhancements

Planned features:
- **Enhanced Summarization Models** - Dedicated summarization providers for better compression
- **Multi-modal Support** - Image and document upload capabilities
- **Advanced Analytics** - Detailed conversation insights and usage patterns
- **Custom Provider Integration** - Plugin system for additional AI providers
- **Team Collaboration** - Shared conversations and workspace management
- **Advanced Export Formats** - PDF, Markdown, and other export options
- **Conversation Search** - Full-text search across conversation history
- **Theme Customization** - Dark mode and custom UI themes

---

**Built with ❤️ using Streamlit, Python, and multiple AI providers**

*Enjoy unlimited conversations with smart context management!*
