import { useState, useRef, useEffect } from 'react';
import { Send, FileText, MessageCircle, Loader2, Table, BarChart3, FileStack } from 'lucide-react';
import { Document, Message } from '../App';

interface DocumentChatProps {
  selectedDocument: Document | null;
  messages: Message[];
  onSendMessage: (text: string) => void;
  onOpenGeneralChat: () => void;
  isGeneralChatOpen: boolean;
  isLoading: boolean;
  documentSummary: string;
}

export function DocumentChat({
  selectedDocument,
  messages,
  onSendMessage,
  onOpenGeneralChat,
  isGeneralChatOpen,
  isLoading,
  documentSummary,
}: DocumentChatProps) {
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim() && selectedDocument && !isLoading) {
      onSendMessage(inputValue.trim());
      setInputValue('');
    }
  };

  const renderSource = (source: any) => {
    if (source.type === 'table') {
      return (
        <div className="flex items-center gap-1.5 text-xs bg-blue-100 text-blue-800 px-2.5 py-1.5 rounded-md shadow-sm">
          <Table className="w-3.5 h-3.5" />
          <span className="font-medium">
            {source.table_name || `Table ${source.table_id}`}
          </span>
          <span className="text-blue-600">•</span>
          <span className="text-blue-700">Page {source.page}</span>
        </div>
      );
    } else if (source.type === 'chart') {
      return (
        <div className="flex items-center gap-1.5 text-xs bg-purple-100 text-purple-800 px-2.5 py-1.5 rounded-md shadow-sm">
          <BarChart3 className="w-3.5 h-3.5" />
          <span className="font-medium">
            {source.chart_name || `Chart ${source.chart_id}`}
          </span>
          <span className="text-purple-600">•</span>
          <span className="text-purple-700">Page {source.page}</span>
        </div>
      );
    } else {
      return (
        <div className="flex items-center gap-1.5 text-xs bg-gray-100 text-gray-700 px-2.5 py-1.5 rounded-md shadow-sm">
          <FileText className="w-3.5 h-3.5" />
          <span className="font-medium">Page {source.page}</span>
        </div>
      );
    }
  };

  return (
    <div className="w-full h-full flex flex-col bg-white">
      {/* Header */}
      <div className="p-4 lg:p-6 border-b border-gray-200 bg-white flex-shrink-0">
        {selectedDocument ? (
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
                  <FileText className="w-5 h-5 text-blue-600" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-gray-800">Document Analysis</h2>
                  <p className="text-sm text-gray-500 truncate max-w-xs">{selectedDocument.name}</p>
                </div>
              </div>
              <button
                onClick={onOpenGeneralChat}
                disabled={isGeneralChatOpen}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2 shadow-sm hover:shadow"
              >
                <MessageCircle className="w-5 h-5" />
                <span className="hidden sm:inline">Internet Search</span>
              </button>
            </div>
            
            {/* Document Features */}
            {(selectedDocument.has_tables || selectedDocument.has_charts) && (
              <div className="flex items-center gap-2 mb-3">
                <FileStack className="w-4 h-4 text-gray-400" />
                <div className="flex items-center gap-2">
                  {selectedDocument.has_tables && (
                    <span className="text-xs bg-blue-50 text-blue-700 px-2 py-1 rounded">
                      Contains Tables
                    </span>
                  )}
                  {selectedDocument.has_charts && (
                    <span className="text-xs bg-purple-50 text-purple-700 px-2 py-1 rounded">
                      Contains Charts
                    </span>
                  )}
                </div>
              </div>
            )}
            
            {/* Document Summary */}
            {documentSummary && selectedDocument && (
              <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-xs text-blue-800">
                  <strong>Status:</strong> {documentSummary}
                </p>
              </div>
            )}
          </div>
        ) : (
          <div>
            <h2 className="text-lg font-semibold text-gray-800">Document Q&A</h2>
            <p className="text-sm text-gray-500 mt-1">Select or upload a document to start</p>
          </div>
        )}
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 lg:p-6 bg-gray-50">
        {!selectedDocument ? (
          <div className="h-full flex items-center justify-center text-gray-400">
            <div className="text-center">
              <FileText className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p className="text-lg font-medium text-gray-700">Select or upload a document</p>
              <p className="mt-2 text-sm">to start asking questions</p>
            </div>
          </div>
        ) : messages.length === 0 ? (
          <div className="h-full flex items-center justify-center text-gray-400">
            <div className="text-center max-w-md">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <FileText className="w-8 h-8 text-white" />
              </div>
              <p className="text-lg font-medium text-gray-700 mb-2">Ask questions about this document</p>
              <p className="text-sm text-gray-500 mb-4">I can analyze text, tables, and charts using Gemini Flash + Groq.</p>
              <div className="space-y-2">
                <div className="text-xs text-gray-500 bg-white border border-gray-200 rounded-lg p-3 text-left">
                  <span className="font-semibold">Example:</span> "What does Table 1 show?"
                </div>
                <div className="text-xs text-gray-500 bg-white border border-gray-200 rounded-lg p-3 text-left">
                  <span className="font-semibold">Example:</span> "Summarize the main findings"
                </div>
                <div className="text-xs text-gray-500 bg-white border border-gray-200 rounded-lg p-3 text-left">
                  <span className="font-semibold">Example:</span> "What trends are shown in the charts?"
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-6 max-w-4xl mx-auto">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${
                  message.sender === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                {message.sender === 'bot' && (
                  <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center flex-shrink-0 text-white text-xs font-bold shadow-md">
                    AI
                  </div>
                )}
                <div className="max-w-[75%]">
                  <div
                    className={`rounded-2xl px-4 py-3 shadow-md ${
                      message.sender === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-white text-gray-900 border border-gray-200'
                    }`}
                  >
                    <p className="leading-relaxed whitespace-pre-wrap">{message.text}</p>
                    
                    {/* Sources */}
                    {message.sources && message.sources.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-200">
                        <p className="text-xs font-semibold mb-2 text-gray-600">📚 Sources:</p>
                        <div className="flex flex-wrap gap-2">
                          {message.sources.map((source, idx) => (
                            <div key={idx}>{renderSource(source)}</div>
                          ))}
                        </div>
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
            ))}
            {isLoading && (
              <div className="flex gap-3 justify-start">
                <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center flex-shrink-0 text-white text-xs font-bold shadow-md">
                  AI
                </div>
                <div className="bg-white text-gray-900 border border-gray-200 rounded-2xl px-4 py-3 shadow-md">
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
                    <span className="text-sm text-gray-500">Analyzing with Gemini + Groq...</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 lg:p-6 border-t border-gray-200 bg-white flex-shrink-0">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex gap-2 lg:gap-3">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={
                selectedDocument
                  ? 'Ask about text, tables, or charts...'
                  : 'Select a document first...'
              }
              disabled={!selectedDocument || isLoading}
              className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed transition-all"
            />
            <button
              type="submit"
              disabled={!selectedDocument || !inputValue.trim() || isLoading}
              className="px-4 lg:px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2 shadow-sm hover:shadow"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
              <span className="hidden lg:inline">Send</span>
            </button>
            <button
              type="button"
              onClick={onOpenGeneralChat}
              disabled={isGeneralChatOpen}
              className="px-4 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
              title="Open General Chat"
            >
              <MessageCircle className="w-5 h-5" />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}