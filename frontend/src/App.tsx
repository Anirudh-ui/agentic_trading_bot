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

  const [documentSessionId, setDocumentSessionId] = useState<string>('');
  const [generalSessionId] = useState(`trade_app_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);

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

  const createSessionForDocument = async (docId: string) => {
    try {
      const resp = await fetch(`${API_BASE_URL}/document-session/create`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_id: docId })
      });

      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Failed to create session");

      return data.session_id;
    } catch (err) {
      console.error("[SESSION CREATE] Error:", err);
      return null;
    }
  };

  const handleDocumentUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;

    setIsUploading(true);
    setUploadProgress('Uploading document...');

    try {
      const file = files[0];
      const maxSize = 50 * 1024 * 1024;
      if (file.size > maxSize) {
        alert('File size exceeds 50MB limit');
        setIsUploading(false);
        return;
      }

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
        const e = await response.json();
        throw new Error(e.detail || "Upload failed");
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

      const exists = documents.find(d => d.id === newDocument.id);
      if (!exists) setDocuments(prev => [...prev, newDocument]);

      setSelectedDocument(newDocument);
      setDocumentMessages([]);

      const session = await createSessionForDocument(newDocument.id);
      setDocumentSessionId(session);

      if (uploadedDoc.is_first_time) {
        const features = [];
        if (uploadedDoc.has_tables) features.push('tables');
        if (uploadedDoc.has_charts) features.push('charts');
        const f = features.length ? ` Contains: ${features.join(', ')}.` : '';
        setDocumentSummary(`Document uploaded successfully (${uploadedDoc.page_count} pages).${f}`);
      } else {
        setDocumentSummary(uploadedDoc.previous_summary || "Resuming previous conversation.");
      }

      setUploadProgress('');
    } catch (err: any) {
      console.error("[UPLOAD] Error:", err);
      alert(err.message || "Upload error");
      setUploadProgress('');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDocumentSelect = async (doc: Document | null) => {
    if (isDocLoading) return;

    if (!doc) {
      setSelectedDocument(null);
      setDocumentMessages([]);
      setDocumentSummary('');
      setDocumentSessionId('');
      return;
    }

    setSelectedDocument(doc);
    setDocumentMessages([]);

    const sessionId = await createSessionForDocument(doc.id);
    setDocumentSessionId(sessionId);

    const resp = await fetch(`${API_BASE_URL}/document-history/${doc.id}`);
    const data = await resp.json();

    if (!data.has_history) {
      setDocumentSummary("You are interacting with this document for the first time.");
      return;
    }

    setDocumentSummary(data.summary || "Previous summary available");
  };

  // -----------------------------------------------------
  // 🔥🔥🔥 FIXED RAG BUG — FAST PATH HANDLING ADDED HERE
  // -----------------------------------------------------
  const handleSendDocumentMessage = async (text: string) => {
    if (!selectedDocument || isDocLoading) return;

    const userMsg: Message = {
      id: `user_${Date.now()}`,
      text,
      sender: "user",
      timestamp: new Date()
    };

    setDocumentMessages(prev => [...prev, userMsg]);
    setIsDocLoading(true);

    try {
      if (!documentSessionId) {
        setDocumentMessages(prev => [
          ...prev,
          {
            id: `error_${Date.now()}`,
            text: "No document session found. Re-select the document.",
            sender: "bot",
            timestamp: new Date()
          }
        ]);
        setIsDocLoading(false);
        return;
      }

      // ----------------------------
      // CALL BACKEND
      // ----------------------------
      const resp = await fetch(`${API_BASE_URL}/document-query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: text,
          session_id: documentSessionId,
          doc_id: selectedDocument.id
        })
      });

      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "Query failed");

      // -------------------------------------------------------
      // 🔥🔥🔥 FAST PATH — GREETING, NOT_RELATED, SUMMARY, HISTORY
      // -------------------------------------------------------
      if (data.is_fast_path === true) {
        const botMessage: Message = {
          id: data.message_id,
          text: data.answer,
          sender: "bot",
          timestamp: new Date(data.timestamp),
          sources: [],             // 🔥 NO SOURCES FOR FAST-PATH
          query_type: data.query_type,
          is_fast_path: true
        };

        setDocumentMessages(prev => [...prev, botMessage]);

        if (data.doc_summary) setDocumentSummary(data.doc_summary);

        console.log("[FAST-PATH] handled:", data.query_type);

        setIsDocLoading(false);
        return;  // 🔥 IMPORTANT
      }

      // -------------------------------------------------------
      // 🔵 NORMAL RAG FLOW (DOCUMENT_QUERY)
      // -------------------------------------------------------
      const botMessage: Message = {
        id: data.message_id,
        text: data.answer,
        sender: "bot",
        timestamp: new Date(data.timestamp),
        sources: data.sources || [],
        query_type: data.query_type,
        is_fast_path: false
      };

      setDocumentMessages(prev => [...prev, botMessage]);

      if (data.doc_summary) setDocumentSummary(data.doc_summary);

    } catch (err: any) {
      setDocumentMessages(prev => [
        ...prev,
        {
          id: `error_${Date.now()}`,
          text: err.message || "Something went wrong. Try again.",
          sender: "bot",
          timestamp: new Date()
        }
      ]);
    } finally {
      setIsDocLoading(false);
    }
  };

  // -----------------------------------------------------
  // GENERAL CHAT HANDLER (unchanged)
  // -----------------------------------------------------
  const handleSendGeneralMessage = async (text: string) => {
    if (isGenLoading) return;
    
    const userMsg: Message = {
      id: `user_${Date.now()}`,
      text,
      sender: "user",
      timestamp: new Date()
    };

    setGeneralMessages(prev => [...prev, userMsg]);
    setIsGenLoading(true);

    try {
      const resp = await fetch(`${API_BASE_URL}/internet-query`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          question: text,
          session_id: generalSessionId
        })
      });

      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "Query failed");

      const botMsg: Message = {
        id: data.message_id,
        text: data.answer,
        sender: "bot",
        timestamp: new Date(data.timestamp),
        sources: data.sources || [],
        is_fast_path: data.is_fast_path
      };

      setGeneralMessages(prev => [...prev, botMsg]);

    } catch (err: any) {
      setGeneralMessages(prev => [
        ...prev,
        {
          id: `error_${Date.now()}`,
          text: err.message || "Error occurred.",
          sender: "bot",
          timestamp: new Date()
        }
      ]);
    } finally {
      setIsGenLoading(false);
    }
  };

  const handleDeleteDocument = async (docId: string) => {
    if (!confirm("Delete document permanently?")) return;

    try {
      await fetch(`${API_BASE_URL}/delete-document/${docId}`, {
        method: "DELETE"
      });

      setDocuments(prev => prev.filter(d => d.id !== docId));

      if (selectedDocument?.id === docId) {
        setSelectedDocument(null);
        setDocumentMessages([]);
        setDocumentSummary('');
      }
    } catch (err) {
      alert("Delete failed.");
    }
  };

  const handleCloseGeneralChat = async () => {
    setIsClosing(true);

    try {
      await fetch(`${API_BASE_URL}/clear-internet-session?session_id=${generalSessionId}`, {
        method: "POST"
      });
    } catch {}

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
          className="fixed inset-0 bg-black bg-opacity-40 z-30"
          onClick={handleCloseGeneralChat}
        />
      )}

      {/* Sidebar */}
      <div className="w-[30%] relative z-40">
        <DocumentSidebar
          documents={documents}
          selectedDocument={selectedDocument}
          onDocumentUpload={handleDocumentUpload}
          onDocumentSelect={handleDocumentSelect}
          onDocumentDelete={handleDeleteDocument}
          isUploading={isUploading}
          uploadProgress={uploadProgress}
        />
      </div>

      {/* Document Chat */}
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

      {/* General Chat */}
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
