import { useState } from 'react';
import { DocumentSidebar } from './components/DocumentSidebar';
import { DocumentChat } from './components/DocumentChat';
import { GeneralChatPanel } from './components/GeneralChatPanel';
import { MessageCircle } from 'lucide-react';

export interface Document {
  id: string;
  name: string;
  uploadedAt: Date;
  size: string;
}

export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

export default function App() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [documentMessages, setDocumentMessages] = useState<Message[]>([]);
  const [generalMessages, setGeneralMessages] = useState<Message[]>([]);
  const [isGeneralChatOpen, setIsGeneralChatOpen] = useState(false);
  const [isClosing, setIsClosing] = useState(false);

  const handleDocumentUpload = (files: FileList | null) => {
    if (!files) return;

    const newDocuments: Document[] = Array.from(files).map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      uploadedAt: new Date(),
      size: `${(file.size / 1024).toFixed(1)} KB`,
    }));

    setDocuments(prev => [...prev, ...newDocuments]);
    
    // Auto-select the first uploaded document if none is selected
    if (!selectedDocument && newDocuments.length > 0) {
      setSelectedDocument(newDocuments[0]);
    }
  };

  const handleDocumentSelect = (doc: Document) => {
    setSelectedDocument(doc);
  };

  const handleSendDocumentMessage = (text: string) => {
    const userMessage: Message = {
      id: Math.random().toString(36).substr(2, 9),
      text,
      sender: 'user',
      timestamp: new Date(),
    };

    setDocumentMessages(prev => [...prev, userMessage]);

    // Simulate bot response
    setTimeout(() => {
      const botMessage: Message = {
        id: Math.random().toString(36).substr(2, 9),
        text: `Based on the document "${selectedDocument?.name}", here's the answer to your question. This is a simulated response.`,
        sender: 'bot',
        timestamp: new Date(),
      };
      setDocumentMessages(prev => [...prev, botMessage]);
    }, 1000);
  };

  const handleSendGeneralMessage = (text: string) => {
    const userMessage: Message = {
      id: Math.random().toString(36).substr(2, 9),
      text,
      sender: 'user',
      timestamp: new Date(),
    };

    setGeneralMessages(prev => [...prev, userMessage]);

    // Simulate bot response
    setTimeout(() => {
      const botMessage: Message = {
        id: Math.random().toString(36).substr(2, 9),
        text: `This is a general response to your query. I can help with various topics!`,
        sender: 'bot',
        timestamp: new Date(),
      };
      setGeneralMessages(prev => [...prev, botMessage]);
    }, 1000);
  };

  const handleCloseGeneralChat = () => {
    setIsClosing(true);
    // Wait for animation to complete before clearing messages and hiding
    setTimeout(() => {
      setIsGeneralChatOpen(false);
      setGeneralMessages([]);
      setIsClosing(false);
    }, 300); // Match animation duration
  };

  return (
    <div className="flex h-screen bg-gray-50 relative overflow-hidden">
      {/* Darkened overlay when general chat is open */}
      {isGeneralChatOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-40 z-30 transition-opacity duration-300" 
          onClick={handleCloseGeneralChat}
        />
      )}

      {/* Left Panel - Documents (30%) */}
      <div className="w-[30%] relative z-40">
        <DocumentSidebar
          documents={documents}
          selectedDocument={selectedDocument}
          onDocumentUpload={handleDocumentUpload}
          onDocumentSelect={handleDocumentSelect}
        />
      </div>

      {/* Middle Panel - Document Chat (70% or adjusted) */}
      <div className="flex-1 relative z-40">
        <DocumentChat
          selectedDocument={selectedDocument}
          messages={documentMessages}
          onSendMessage={handleSendDocumentMessage}
          onOpenGeneralChat={() => setIsGeneralChatOpen(true)}
          isGeneralChatOpen={isGeneralChatOpen}
        />
      </div>

      {/* General Chat Panel - expands/collapses from right */}
      {(isGeneralChatOpen || isClosing) && (
        <div className={`relative z-40 ${isClosing ? 'animate-collapse' : 'animate-expand'}`} style={{ width: isClosing ? '320px' : '320px' }}>
          <GeneralChatPanel
            isOpen={isGeneralChatOpen}
            messages={generalMessages}
            onClose={handleCloseGeneralChat}
            onSendMessage={handleSendGeneralMessage}
            isClosing={isClosing}
          />
        </div>
      )}
    </div>
  );
}