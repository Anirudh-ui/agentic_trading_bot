import { Upload, FileText, Loader2, FileCheck, Table, BarChart3 } from 'lucide-react';
import { Document } from '../App';

interface DocumentSidebarProps {
  documents: Document[];
  selectedDocument: Document | null;
  onDocumentUpload: (files: FileList | null) => void;
  onDocumentSelect: (doc: Document) => void;
  isUploading: boolean;
  uploadProgress?: string;
}

export function DocumentSidebar({
  documents,
  selectedDocument,
  onDocumentUpload,
  onDocumentSelect,
  isUploading,
  uploadProgress
}: DocumentSidebarProps) {
  return (
    <div className="w-full h-full bg-white border-r border-gray-200 flex flex-col">
      {/* Header */}
      <div className="p-4 lg:p-6 border-b border-gray-200 flex-shrink-0">
        <h2 className="text-xl font-bold text-gray-800 mb-4">Documents</h2>
        
        {/* Upload Button */}
        <label className={`flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer transition-colors ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}`}>
          {isUploading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span className="text-sm">Processing...</span>
            </>
          ) : (
            <>
              <Upload className="w-5 h-5" />
              <span>Upload Document</span>
            </>
          )}
          <input
            type="file"
            accept=".pdf,.doc,.docx"
            onChange={(e) => onDocumentUpload(e.target.files)}
            disabled={isUploading}
            className="hidden"
          />
        </label>
        
        {/* Upload Progress */}
        {uploadProgress && (
          <div className="mt-3 p-2 bg-blue-50 border border-blue-200 rounded text-xs text-blue-700 flex items-center gap-2">
            <Loader2 className="w-4 h-4 animate-spin" />
            {uploadProgress}
          </div>
        )}
        
        {/* Supported Formats */}
        <p className="mt-2 text-xs text-gray-500">
          Supported: PDF, DOCX (Max 50MB)
        </p>
      </div>

      {/* Document List */}
      <div className="flex-1 overflow-y-auto p-4">
        {documents.length === 0 ? (
          <div className="text-center text-gray-400 mt-8">
            <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p className="font-medium">No documents uploaded</p>
            <p className="mt-1 text-sm">Upload documents to get started</p>
          </div>
        ) : (
          <div className="space-y-2">
            {documents.map((doc) => (
              <div
                key={doc.id}
                onClick={() => onDocumentSelect(doc)}
                className={`p-3 rounded-lg border cursor-pointer transition-all hover:shadow-md ${
                  selectedDocument?.id === doc.id
                    ? 'bg-blue-50 border-blue-300 shadow-sm'
                    : 'bg-white border-gray-200 hover:border-gray-300'
                }`}
              >
                <div className="flex items-start gap-3">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${
                    selectedDocument?.id === doc.id
                      ? 'bg-blue-100'
                      : 'bg-gray-100'
                  }`}>
                    <FileText className={`w-5 h-5 ${
                      selectedDocument?.id === doc.id
                        ? 'text-blue-600'
                        : 'text-gray-600'
                    }`} />
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <p className={`font-medium truncate ${
                      selectedDocument?.id === doc.id
                        ? 'text-blue-900'
                        : 'text-gray-800'
                    }`}>
                      {doc.name}
                    </p>
                    
                    <div className="flex items-center gap-2 mt-1">
                      <p className="text-xs text-gray-500">
                        {doc.page_count ? `${doc.page_count} pages` : doc.size}
                      </p>
                      
                      {/* Features badges */}
                      <div className="flex items-center gap-1">
                        {doc.has_tables && (
                          <div className="flex items-center gap-0.5 px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">
                            <Table className="w-3 h-3" />
                            <span className="hidden sm:inline">Tables</span>
                          </div>
                        )}
                        {doc.has_charts && (
                          <div className="flex items-center gap-0.5 px-1.5 py-0.5 bg-purple-100 text-purple-700 rounded text-xs">
                            <BarChart3 className="w-3 h-3" />
                            <span className="hidden sm:inline">Charts</span>
                          </div>
                        )}
                      </div>
                    </div>
                    
                    <p className="text-xs text-gray-400 mt-1">
                      {new Date(doc.uploadedAt).toLocaleDateString()}
                    </p>
                  </div>
                  
                  {selectedDocument?.id === doc.id && (
                    <div className="flex-shrink-0">
                      <FileCheck className="w-5 h-5 text-blue-600" />
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
      
      {/* Footer Info */}
      <div className="p-4 border-t border-gray-200 flex-shrink-0">
        <div className="text-xs text-gray-500 space-y-1">
          <p className="flex items-center gap-2">
            <span className="font-semibold">{documents.length}</span>
            document{documents.length !== 1 ? 's' : ''} uploaded
          </p>
          <p className="text-gray-400">
            Powered by Gemini Flash
          </p>
        </div>
      </div>
    </div>
  );
}