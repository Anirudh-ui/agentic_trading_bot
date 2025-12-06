CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS documents (
    doc_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_hash TEXT UNIQUE NOT NULL,
    file_type TEXT,
    page_count INT,
    has_tables BOOLEAN DEFAULT FALSE,
    has_charts BOOLEAN DEFAULT FALSE,
    size_bytes BIGINT,
    summary TEXT,
    uploaded_at TIMESTAMP DEFAULT NOW()
);
