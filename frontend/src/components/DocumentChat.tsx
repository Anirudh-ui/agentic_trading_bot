import { useState, useRef, useEffect } from 'react';
import { Send, FileText, MessageCircle } from 'lucide-react';
import { Document, Message } from '../App';

interface DocumentChatProps {
  selectedDocument: Document | null;
  messages: Message[];
  onSendMessage: (text: string) => void;
  onOpenGeneralChat: () => void;
  isGeneralChatOpen: boolean;
}

export function DocumentChat({
  selectedDocument,
  messages,
  onSendMessage,
  onOpenGeneralChat,
  isGeneralChatOpen,
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
    if (inputValue.trim() && selectedDocument) {
      onSendMessage(inputValue.trim());
      setInputValue('');
    }
  };

  return (
    <div className="w-full h-full flex flex-col bg-white">
      {/* Header */}
      <div className="p-4 lg:p-6 border-b border-gray-200 bg-white flex-shrink-0">
        {selectedDocument ? (
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
              <FileText className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <h2>Chat with Document</h2>
              <p className="text-gray-500 mt-1">{selectedDocument.name}</p>
            </div>
          </div>
        ) : (
          <div>
            <h2>Document Q&A</h2>
            <p className="text-gray-500 mt-1">Select a document to start chatting</p>
          </div>
        )}
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 lg:p-6">
        {!selectedDocument ? (
          <div className="h-full flex items-center justify-center text-gray-400">
            <div className="text-center">
              <FileText className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p>Select a document from the sidebar</p>
              <p className="mt-2">to start asking questions</p>
            </div>
          </div>
        ) : messages.length === 0 ? (
          <div className="h-full flex items-center justify-center text-gray-400">
            <div className="text-center">
              <p>Ask questions about the document</p>
              <p className="mt-2">The AI will help you find answers</p>
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
                  <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center flex-shrink-0 text-white">
                    AI
                  </div>
                )}
                <div
                  className={`max-w-[75%] rounded-2xl px-4 py-3 ${
                    message.sender === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-900 border border-gray-200'
                  }`}
                >
                  <p className="leading-relaxed">{message.text}</p>
                  <p
                    className={`mt-2 ${
                      message.sender === 'user' ? 'text-blue-100' : 'text-gray-400'
                    }`}
                  >
                    {message.timestamp.toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </p>
                </div>
                {message.sender === 'user' && (
                  <div className="w-8 h-8 bg-gradient-to-br from-gray-600 to-gray-800 rounded-full flex items-center justify-center flex-shrink-0 text-white">
                    U
                  </div>
                )}
              </div>
            ))}
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
                  ? 'Ask a question about the document...'
                  : 'Select a document first...'
              }
              disabled={!selectedDocument}
              className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed transition-all"
            />
            <button
              type="submit"
              disabled={!selectedDocument || !inputValue.trim()}
              className="px-4 lg:px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              <Send className="w-5 h-5" />
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