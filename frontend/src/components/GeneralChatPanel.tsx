import { useState, useRef, useEffect, useCallback } from 'react';
import { X, Send, MessageCircle, Loader2, Zap, TrendingUp, Newspaper, Sparkles } from 'lucide-react';
import { Message } from '../App';

interface GeneralChatPanelProps {
  isOpen: boolean;
  messages: Message[];
  onClose: () => void;
  onSendMessage: (text: string) => void;
  isClosing: boolean;
  isLoading: boolean;
}

// REFACTORED UTILITY FUNCTION: Converts messy markdown to clean HTML with clickable links (CTAs)
const convertMarkdownToHtml = (markdownText: string): string => {
    let htmlText = markdownText;

    // 1. Convert markdown links [text](url) to <a> tags (CTA). 
    // Uses <span> with 'block' and margin to make them look like distinct CTAs at the end.
    htmlText = htmlText.replace(
        /\[([^\]]+)\]\(([^)]+)\)/g, 
        '<span class="block mt-2 text-sm"><a href="$2" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:text-blue-700 underline font-medium">$1</a></span>'
    );
    
    // 2. Aggressively remove all bolding and emphasis markers (**, *, _, ~)
    // This targets both single (*, _) and double (**, __) markers globally.
    // Requirement: "without stars and astrix"
    htmlText = htmlText.replace(/[\*\_\~]+/g, '');

    // 3. Convert H3 Headers (###) to a clean, bold, separated section title. 
    // Requirement: Make it crisp and good looking
    htmlText = htmlText.replace(
        /### ([^\n]+)/g, 
        '<br/><span class="block text-base font-semibold text-gray-900 mt-3 mb-1">$1</span>'
    );
    
    // 4. Handle List markers:
    // a. Numbered lists (e.g., 1. text) - adds a space and makes the number bold for separation
    htmlText = htmlText.replace(/^(\d+\. )/gm, '<br/><span class="font-medium">$1</span>');
    // b. Bullet points ( - ) (Must handle the space before and after for list formatting)
    htmlText = htmlText.replace(/(\n)[\s]*(- )/g, '<br/>&nbsp;&nbsp;&nbsp;&nbsp;&bull; ');

    // 5. Clean up the horizontal rule (---)
    htmlText = htmlText.replace(/---/g, '<br/><hr class="my-2 border-t border-gray-300"/><br/>');

    // 6. Convert explicit newlines to <br/> tags to maintain structure, including paragraphs and list separation
    htmlText = htmlText.replace(/\n/g, '<br/>');

    // 7. Remove any leading/trailing <br/> left over from processing for a clean start/end
    htmlText = htmlText.replace(/^(<br\/>)*/, '').replace(/(<br\/>)*$/, '');

    // 8. Condense multiple breaks into single breaks for clean paragraph separation
    htmlText = htmlText.replace(/(<br\/>\s*){3,}/g, '<br/><br/>');

    return htmlText;
};


export function GeneralChatPanel({
  isOpen,
  messages,
  onClose,
  onSendMessage,
  isClosing,
  isLoading,
}: GeneralChatPanelProps) {
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Auto-focus input when opened
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim() && !isLoading) {
      onSendMessage(inputValue.trim());
      setInputValue('');
      setIsTyping(false);
    }
  }, [inputValue, isLoading, onSendMessage]);

  // Quick action suggestions with optimized queries
  const quickActions = [
    { 
      text: "What's AAPL stock price?", 
      icon: TrendingUp,
      category: 'stock',
      color: 'bg-blue-50 hover:bg-blue-100 text-blue-700'
    },
    { 
      text: "Latest Tesla news", 
      icon: Newspaper,
      category: 'news',
      color: 'bg-purple-50 hover:bg-purple-100 text-purple-700'
    },
    { 
      text: "How is MSFT doing?", 
      icon: TrendingUp,
      category: 'stock',
      color: 'bg-green-50 hover:bg-green-100 text-green-700'
    },
    { 
      text: "Market trends today", 
      icon: Newspaper,
      category: 'news',
      color: 'bg-orange-50 hover:bg-orange-100 text-orange-700'
    },
  ];

  const getQueryTypeIcon = (queryType?: string, isFastPath?: boolean) => {
    if (isFastPath) return <Zap className="w-3.5 h-3.5 text-yellow-600" />;
    if (queryType === 'stock') return <TrendingUp className="w-3.5 h-3.5 text-blue-600" />;
    if (queryType === 'news') return <Newspaper className="w-3.5 h-3.5 text-purple-600" />;
    return <MessageCircle className="w-3.5 h-3.5 text-gray-500" />;
  };

  const getQueryTypeBadge = (queryType?: string, isFastPath?: boolean) => {
    if (isFastPath) {
      return (
        <div className="flex items-center gap-1 text-xs bg-yellow-100 text-yellow-800 px-2 py-0.5 rounded-full">
          <Zap className="w-3 h-3" />
          <span className="font-medium">Instant</span>
        </div>
      );
    }
    
    if (queryType === 'stock') {
      return (
        <div className="flex items-center gap-1 text-xs bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full">
          <TrendingUp className="w-3 h-3" />
          <span className="font-medium">Stock</span>
        </div>
      );
    }
    
    if (queryType === 'news') {
      return (
        <div className="flex items-center gap-1 text-xs bg-purple-100 text-purple-800 px-2 py-0.5 rounded-full">
          <Newspaper className="w-3 h-3" />
          <span className="font-medium">News</span>
        </div>
      );
    }
    
    return null;
  };

  return (
    <div className="h-full bg-white shadow-2xl flex flex-col border-l border-gray-200">
      {/* Optimized Header */}
      <div className="p-4 border-b border-gray-200 flex items-center justify-between flex-shrink-0 bg-gradient-to-r from-blue-50 via-purple-50 to-green-50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-600 via-purple-600 to-green-600 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-base font-bold text-gray-900">Financial Chat</h2>
            <p className="text-xs text-gray-600 mt-0.5">Powered by AI</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-2 hover:bg-white rounded-lg transition-all"
          aria-label="Close chat"
        >
          <X className="w-5 h-5 text-gray-600" />
        </button>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 bg-gradient-to-b from-gray-50 to-white">
        {messages.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center max-w-sm">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-600 via-purple-600 to-green-600 rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg">
                <Sparkles className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-lg font-bold text-gray-800 mb-2">Welcome!</h3>
              <p className="text-sm text-gray-600 mb-1">Ask about stocks, news, or market trends</p>
              <p className="text-xs text-gray-500">Lightning-fast responses ⚡</p>
              
              {/* Quick Action Buttons */}
              <div className="mt-6 space-y-2">
                <div className="text-xs text-gray-600 mb-3 font-semibold">💡 Quick Actions:</div>
                {quickActions.map((action, idx) => {
                  const Icon = action.icon;
                  return (
                    <button
                      key={idx}
                      onClick={() => setInputValue(action.text)}
                      className={`w-full text-left p-2.5 text-xs rounded-lg border border-gray-200 hover:shadow-sm cursor-pointer transition-all flex items-center gap-2 ${action.color}`}
                    >
                      <Icon className="w-4 h-4" />
                      <span className="font-medium">{action.text}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((message) => {
              
              // Detect if this is the specific static response that requires aggressive formatting
              // Check for the presence of the section titles or links typically generated by the internet workflow.
              const isStaticResponse = message.sender === 'bot' && 
                                       (message.text.includes('###') || message.text.includes('http')); 

              return (
                <div
                  key={message.id}
                  className={`flex gap-2.5 ${
                    message.sender === 'user' ? 'justify-end' : 'justify-start'
                  }`}
                >
                  {message.sender === 'bot' && (
                    <div className="w-8 h-8 bg-gradient-to-br from-blue-600 via-purple-600 to-green-600 rounded-full flex items-center justify-center flex-shrink-0 text-white text-xs font-bold shadow-md">
                      AI
                    </div>
                  )}
                  {/* Message is rendered inside the message card (max-w-[85%]) */}
                  <div className="max-w-[85%]">
                    <div
                      className={`rounded-2xl px-4 py-2.5 shadow-md ${
                        message.sender === 'user'
                          ? 'bg-gradient-to-br from-blue-600 to-blue-700 text-white'
                          : 'bg-white text-gray-900 border border-gray-200'
                      }`}
                    >
                      {/* CONDITIONAL RENDERING: Use the parser for the static response, 
                          or the default renderer for everything else. */}
                      {isStaticResponse ? (
                        <p 
                            className="leading-relaxed text-sm break-words" // break-words for URL wrapping
                            dangerouslySetInnerHTML={{ __html: convertMarkdownToHtml(message.text) }}
                        />
                      ) : (
                        // Default rendering for standard chat (including fast-path/stock/news replies)
                        <p className="leading-relaxed text-sm whitespace-pre-wrap">{message.text}</p>
                      )}
                      
                      {/* Query Type Badge */}
                      {message.sender === 'bot' && (message.query_type || message.is_fast_path) && (
                        <div className="mt-2 pt-2 border-t border-gray-200 flex items-center gap-2">
                          {getQueryTypeBadge(message.query_type, message.is_fast_path)}
                        </div>
                      )}
                      
                      <p
                        className={`text-xs mt-2 ${
                          message.sender === 'user' ? 'text-blue-100' : 'text-gray-400'
                        }`}
                      >
                        {message.timestamp.toLocaleTimeString([], {
                          hour: '2-digit',
                          minute: '2-digit',
                        })}
                      </p>
                    </div>
                  </div>
                  {message.sender === 'user' && (
                    <div className="w-8 h-8 bg-gradient-to-br from-gray-600 to-gray-800 rounded-full flex items-center justify-center flex-shrink-0 text-white text-xs font-bold shadow-md">
                      U
                    </div>
                  )}
                </div>
              );
            })}
            
            {/* Loading Indicator */}
            {isLoading && (
              <div className="flex gap-2.5 justify-start">
                <div className="w-8 h-8 bg-gradient-to-br from-blue-600 via-purple-600 to-green-600 rounded-full flex items-center justify-center flex-shrink-0 text-white text-xs font-bold shadow-md">
                  AI
                </div>
                <div className="bg-white text-gray-900 border border-gray-200 rounded-2xl px-4 py-2.5 shadow-md">
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
                    <span className="text-sm text-gray-600">Processing...</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Optimized Input Area */}
      <div className="p-4 border-t border-gray-200 bg-white flex-shrink-0">
        {/* Typing Indicator */}
        {isTyping && inputValue && (
          <div className="mb-2 text-xs text-gray-500 flex items-center gap-1">
            <span className="w-1 h-1 bg-gray-400 rounded-full animate-pulse"></span>
            <span>Typing...</span>
          </div>
        )}
        
        <form onSubmit={handleSubmit}>
          <div className="flex gap-2">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => {
                setInputValue(e.target.value);
                setIsTyping(e.target.value.length > 0);
              }}
              placeholder="Ask about stocks, news, trends..."
              disabled={isLoading}
              className="flex-1 px-4 py-2.5 text-sm border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all disabled:bg-gray-100 disabled:cursor-not-allowed"
              autoComplete="off"
            />
            <button
              type="submit"
              disabled={!inputValue.trim() || isLoading}
              className="px-4 py-2.5 bg-gradient-to-r from-blue-600 via-purple-600 to-green-600 text-white rounded-xl hover:shadow-lg disabled:from-gray-300 disabled:to-gray-300 disabled:cursor-not-allowed transition-all flex items-center gap-2"
              aria-label="Send message"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  <span className="text-xs font-medium hidden sm:inline">Send</span>
                </>
              )}
            </button>
          </div>
        </form>
        
        {/* Performance Hint */}
        <div className="mt-2 text-xs text-gray-500 flex items-center gap-1">
          <Zap className="w-3 h-3 text-yellow-500" />
          <span>Instant responses for greetings</span>
        </div>
      </div>
    </div>
  );
}