# 🧠 AgentAI Smart Context Chatbot

A sophisticated multi-provider AI chatbot with intelligent context management and unlimited conversation history through automatic summarization.

## ✨ Features

### 🎯 Multi-Provider Support
- **OpenAI** - GPT-4, GPT-4 Turbo, GPT-3.5 Turbo models
- **Anthropic** - Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku
- **xAI (Grok)** - Grok Beta, Grok Vision Beta
- **Groq** - Llama 3.1 70B/8B, Mixtral 8x7B, Gemma2 9B (Fast inference)
- **Together AI** - Llama 3 70B/8B, Mixtral 8x7B, Nous Hermes models
- **Hugging Face** - DialoGPT, BlenderBot, Llama 2 models
- **OpenRouter** - Unified access to GPT-4, Claude, Llama, Gemini, Mixtral
- **Local/OpenWebUI** - Compatible with Ollama and other local model servers

### 🧠 Smart Context Management
- **Automatic Summarization** - Intelligently summarizes conversations when they get too long
- **Token Optimization** - Maintains conversation flow while minimizing API costs
- **Context Preservation** - Key points and context are preserved across summarizations
- **Unlimited History** - Effectively unlimited conversation length through smart compression

### 💾 Conversation Management
- **Export/Import** - Save and load conversations in JSON format
- **Multiple Conversations** - Manage multiple conversation threads
- **History Tracking** - Browse and reload previous conversations
- **Context Summaries** - View summarized context for long conversations

### ⚙️ Advanced Configuration
- **Model Parameters** - Adjust temperature, max tokens, and system prompts
- **Provider Settings** - Configure API keys and custom endpoints
- **Token Tracking** - Real-time token usage and cost estimation
- **Smart Thresholds** - Configurable summarization triggers

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd claude-client
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
- **Auto-Summarize**: Toggle automatic context summarization
- **Token Limits**: Set maximum context tokens before summarization
- **View Summaries**: Expand context summary sections to see compressed history

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
1. **Token Monitoring** - Tracks conversation length in tokens
2. **Summarization Trigger** - Activates when 70% of token limit is reached
3. **Context Compression** - Creates concise summaries with key points
4. **History Preservation** - Maintains recent messages + summarized context

## 📊 Token Usage & Costs

The application provides real-time tracking of:
- **Input Tokens** - Tokens sent to the API
- **Output Tokens** - Tokens received from the API  
- **Total Usage** - Combined token consumption
- **Estimated Costs** - Approximate API charges with transparent pricing

### Cost Transparency
Cost estimates are based on published API pricing as of 2024:
- **OpenAI GPT-4o**: $0.0025/1K input, $0.01/1K output tokens
- **Anthropic Claude 3.5**: $0.003/1K input, $0.015/1K output tokens  
- **xAI Grok**: ~$0.05/1K tokens (varies by model)
- **Groq**: Free for most models
- **TogetherAI**: ~$0.0008/1K tokens (varies by model)
- **OpenRouter**: ~$0.02/1K tokens (varies significantly by model)

> ⚠️ **Important**: Cost estimates are approximate and may vary from actual API pricing. Always check current provider pricing for accurate costs.

### Cost Optimization
- **Smart Summarization** - Reduces token usage for long conversations
- **Provider Comparison** - Easy switching between cost-effective providers
- **Usage Monitoring** - Track spending across conversations

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
- Disable auto-summarize if experiencing problems
- Manually start new conversations for fresh context
- Check token limits in provider settings

### Performance Tips
- Use Groq for fastest inference (free tier available)
- Enable auto-summarization for long conversations
- Choose smaller models for faster responses
- Use local providers for privacy and speed

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
- **Advanced Summarization** - Using dedicated summarization models
- **Multi-modal Support** - Image and file upload capabilities
- **Plugin System** - Custom provider integrations
- **Conversation Analytics** - Usage statistics and insights
- **Team Collaboration** - Shared conversations and workspaces

---

**Built with ❤️ using Streamlit, Python, and multiple AI providers**

*Enjoy unlimited conversations with smart context management!*
