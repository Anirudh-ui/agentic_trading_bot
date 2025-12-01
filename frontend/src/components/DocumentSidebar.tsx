import { Upload, FileText, Trash2 } from 'lucide-react';
import { Document } from '../App';

interface DocumentSidebarProps {
  documents: Document[];
  selectedDocument: Document | null;
  onDocumentUpload: (files: FileList | null) => void;
  onDocumentSelect: (doc: Document) => void;
}

export function DocumentSidebar({
  documents,
  selectedDocument,
  onDocumentUpload,
  onDocumentSelect,
}: DocumentSidebarProps) {
  return (
    <div className="w-full h-full bg-white border-r border-gray-200 flex flex-col">
      {/* Header */}
      <div className="p-4 lg:p-6 border-b border-gray-200 flex-shrink-0">
        <h2 className="mb-4">Documents</h2>
        
        {/* Upload Button */}
        <label className="flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer transition-colors">
          <Upload className="w-5 h-5" />
          <span>Upload Document</span>
          <input
            type="file"
            multiple
            accept=".pdf,.doc,.docx,.txt"
            onChange={(e) => onDocumentUpload(e.target.files)}
            className="hidden"
          />
        </label>
      </div>

      {/* Document List */}
      <div className="flex-1 overflow-y-auto p-4">
        {documents.length === 0 ? (
          <div className="text-center text-gray-400 mt-8">
            <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No documents uploaded</p>
            <p className="mt-1">Upload documents to get started</p>
          </div>
        ) : (
          <div className="space-y-2">
            {documents.map((doc) => (
              <div
                key={doc.id}
                onClick={() => onDocumentSelect(doc)}
                className={`p-3 rounded-lg border cursor-pointer transition-all ${
                  selectedDocument?.id === doc.id
                    ? 'bg-blue-50 border-blue-300'
                    : 'bg-white border-gray-200 hover:border-gray-300'
                }`}
              >
                <div className="flex items-start gap-3">
                  <FileText className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
                  <div className="flex-1 min-w-0">
                    <p className="truncate">{doc.name}</p>
                    <p className="text-gray-500 mt-1">{doc.size}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}