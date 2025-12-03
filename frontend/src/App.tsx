import { useState, useEffect } from 'react';
import { DocumentSidebar } from './components/DocumentSidebar';
import { DocumentChat } from './components/DocumentChat';
import { GeneralChatPanel } from './components/GeneralChatPanel';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface Document {
  id: string;
  name: string;
  uploadedAt: Date;
  size: string;
  page_count?: number;
  isFirstTime?: boolean;
  previousSummary?: string;
  has_tables?: boolean;
  has_charts?: boolean;
}

export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  sources?: Array<{
    type: string;
    page: number;
    table_id?: string;
    chart_id?: string;
    table_name?: string;
    chart_name?: string;
  }>;
  query_type?: string;
  is_fast_path?: boolean;
}

export default function App() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [documentMessages, setDocumentMessages] = useState<Message[]>([]);
  const [generalMessages, setGeneralMessages] = useState<Message[]>([]);
  const [isGeneralChatOpen, setIsGeneralChatOpen] = useState(false);
  const [isClosing, setIsClosing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isDocLoading, setIsDocLoading] = useState(false);
  const [isGenLoading, setIsGenLoading] = useState(false);
  const [documentSummary, setDocumentSummary] = useState<string>('');
  const [uploadProgress, setUploadProgress] = useState<string>('');
  
  // Session IDs
  const [documentSessionId, setDocumentSessionId] = useState<string>('');
  const [generalSessionId] = useState(`trade_app_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);

  // Load existing documents on mount
  useEffect(() => {
    loadUserDocuments();
  }, []);

  const loadUserDocuments = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/get-user-documents`);
      if (response.ok) {
        const data = await response.json();
        const docs: Document[] = data.documents.map((doc: any) => ({
          id: doc.id,
          name: doc.name,
          uploadedAt: new Date(doc.uploadedAt),
          size: doc.size,
          page_count: doc.page_count || 0,
        }));
        setDocuments(docs);
        console.log('[LOAD DOCS] Loaded:', docs.length);
      }
    } catch (error) {
      console.error('[LOAD DOCS] Error:', error);
    }
  };

  const handleDocumentUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;

    setIsUploading(true);
    setUploadProgress('Uploading document...');
    
    try {
      const file = files[0];
      
      // Validate file size (max 50MB)
      const maxSize = 50 * 1024 * 1024;
      if (file.size > maxSize) {
        alert('File size exceeds 50MB limit');
        setIsUploading(false);
        return;
      }
      
      // Validate file type
      const allowedTypes = ['.pdf', '.docx', '.doc'];
      const fileExt = '.' + file.name.split('.').pop()?.toLowerCase();
      if (!allowedTypes.includes(fileExt)) {
        alert('Only PDF and DOCX files are supported');
        setIsUploading(false);
        return;
      }
      
      const formData = new FormData();
      formData.append('file', file);

      setUploadProgress('Processing with Gemini...');
      
      const response = await fetch(`${API_BASE_URL}/upload-document`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const uploadedDoc = await response.json();
      
      const newDocument: Document = {
        id: uploadedDoc.doc_id,
        name: uploadedDoc.name,
        uploadedAt: new Date(uploadedDoc.uploadedAt),
        size: uploadedDoc.size,
        page_count: uploadedDoc.page_count,
        isFirstTime: uploadedDoc.is_first_time,
        previousSummary: uploadedDoc.previous_summary,
        has_tables: uploadedDoc.has_tables,
        has_charts: uploadedDoc.has_charts
      };

      // Check if document already exists in list
      const existingIndex = documents.findIndex(d => d.id === newDocument.id);
      if (existingIndex === -1) {
        setDocuments(prev => [...prev, newDocument]);
      }
      
      // Auto-select
      setSelectedDocument(newDocument);
      setDocumentMessages([]);
      setDocumentSessionId(newDocument.id); // Session ID = doc_id
      
      // Show document info
      if (uploadedDoc.is_first_time) {
        const features = [];
        if (uploadedDoc.has_tables) features.push('tables');
        if (uploadedDoc.has_charts) features.push('charts');
        const featureText = features.length > 0 ? ` Contains: ${features.join(', ')}.` : '';
        setDocumentSummary(`Document uploaded successfully (${uploadedDoc.page_count} pages).${featureText} Ready to answer questions!`);
      } else {
        setDocumentSummary(uploadedDoc.previous_summary || 'Resuming previous conversation with this document.');
      }
      
      console.log('[UPLOAD] Success:', newDocument);
      setUploadProgress('');
    } catch (error) {
      console.error('[UPLOAD] Error:', error);
      alert(error instanceof Error ? error.message : 'Upload failed. Please try again.');
      setUploadProgress('');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDocumentSelect = async (doc: Document) => {
    if (isDocLoading) return; // Prevent selection during loading
    
    setSelectedDocument(doc);
    setDocumentMessages([]);
    setDocumentSessionId(doc.id);
    
    console.log('[SELECT] Document:', doc.id);
    
    // Load document summary
    try {
      const response = await fetch(`${API_BASE_URL}/get-document-summary/${doc.id}`);
      if (response.ok) {
        const data = await response.json();
        if (data.has_history) {
          setDocumentSummary(data.summary || 'Previous conversations available');
        } else {
          setDocumentSummary('No previous conversations. Ask your first question!');
        }
      }
    } catch (error) {
      console.error('[SUMMARY] Error:', error);
      setDocumentSummary('Ready to answer questions about this document.');
    }
  };

  const handleSendDocumentMessage = async (text: string) => {
    if (!selectedDocument || isDocLoading) return;

    const userMessage: Message = {
      id: `user_${Date.now()}`,
      text,
      sender: 'user',
      timestamp: new Date(),
    };

    setDocumentMessages(prev => [...prev, userMessage]);
    setIsDocLoading(true);

    try {
      console.log('[DOC QUERY] Sending:', text.substring(0, 50));
      
      const response = await fetch(`${API_BASE_URL}/document-query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: text,
          session_id: documentSessionId,
          doc_id: selectedDocument.id,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Query failed');
      }

      const data = await response.json();

      const botMessage: Message = {
        id: data.message_id,
        text: data.answer,
        sender: 'bot',
        timestamp: new Date(data.timestamp),
        sources: data.sources || [],
        query_type: data.query_type
      };

      setDocumentMessages(prev => [...prev, botMessage]);
      
      // Update summary
      if (data.doc_summary) {
        setDocumentSummary(data.doc_summary);
      }
      
      console.log('[DOC QUERY] Success, sources:', data.sources?.length || 0);
      
    } catch (error) {
      console.error('[DOC QUERY] Error:', error);
      
      const errorMessage: Message = {
        id: `error_${Date.now()}`,
        text: error instanceof Error ? error.message : 'Sorry, I encountered an error. Please try again.',
        sender: 'bot',
        timestamp: new Date(),
      };
      
      setDocumentMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsDocLoading(false);
    }
  };

  const handleSendGeneralMessage = async (text: string) => {
    if (isGenLoading) return;
    
    const userMessage: Message = {
      id: `user_${Date.now()}`,
      text,
      sender: 'user',
      timestamp: new Date(),
    };

    setGeneralMessages(prev => [...prev, userMessage]);
    setIsGenLoading(true);

    try {
      console.log('[GENERAL QUERY] Sending:', text.substring(0, 50));
      
      const response = await fetch(`${API_BASE_URL}/internet-query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: text,
          session_id: generalSessionId,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Query failed');
      }

      const data = await response.json();

      const botMessage: Message = {
        id: data.message_id,
        text: data.answer,
        sender: 'bot',
        timestamp: new Date(data.timestamp),
        sources: data.sources || [],
        query_type: data.query_type,
        is_fast_path: data.is_fast_path
      };

      setGeneralMessages(prev => [...prev, botMessage]);
      
      console.log('[GENERAL QUERY] Success, type:', data.query_type, 'fast-path:', data.is_fast_path);
      
    } catch (error) {
      console.error('[GENERAL QUERY] Error:', error);
      
      const errorMessage: Message = {
        id: `error_${Date.now()}`,
        text: error instanceof Error ? error.message : 'Sorry, I encountered an error. Please try again.',
        sender: 'bot',
        timestamp: new Date(),
      };
      
      setGeneralMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsGenLoading(false);
    }
  };

  const handleCloseGeneralChat = async () => {
    setIsClosing(true);
    
    // Clear session in backend
    try {
      console.log('[CLEAR SESSION] Clearing:', generalSessionId);
      await fetch(`${API_BASE_URL}/clear-internet-session?session_id=${generalSessionId}`, {
        method: 'POST'
      });
    } catch (error) {
      console.error('[CLEAR SESSION] Error:', error);
    }
    
    setTimeout(() => {
      setIsGeneralChatOpen(false);
      setGeneralMessages([]);
      setIsClosing(false);
    }, 300);
  };

  return (
    <div className="flex h-screen bg-gray-50 relative overflow-hidden">
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
          isUploading={isUploading}
          uploadProgress={uploadProgress}
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
          isLoading={isDocLoading}
          documentSummary={documentSummary}
        />
      </div>

      {/* General Chat Panel */}
      {(isGeneralChatOpen || isClosing) && (
        <div 
          className={`relative z-40 ${isClosing ? 'animate-collapse' : 'animate-expand'}`} 
          style={{ width: '320px' }}
        >
          <GeneralChatPanel
            isOpen={isGeneralChatOpen}
            messages={generalMessages}
            onClose={handleCloseGeneralChat}
            onSendMessage={handleSendGeneralMessage}
            isClosing={isClosing}
            isLoading={isGenLoading}
          />
        </div>
      )}
    </div>
  );
}