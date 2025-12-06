import psycopg2
import psycopg2.extras
import hashlib

class PostgresManager:
    def __init__(self):
        self.conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="document_ai",
            user="admin",
            password="admin123",
        )
        self.conn.autocommit = True

    @staticmethod
    def compute_hash(file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    def find_by_hash(self, file_hash: str):
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT * FROM documents WHERE file_hash=%s", (file_hash,))
        return cur.fetchone()

    def register_document(
        self,
        user_id: str,
        filename: str,
        file_hash: str,
        file_type: str,
        page_count: int,
        has_tables: bool,
        has_charts: bool,
        size_bytes: int,
    ):
        query = """
        INSERT INTO documents (
            user_id, filename, file_hash, file_type,
            page_count, has_tables, has_charts, size_bytes
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING doc_id, uploaded_at;
        """
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(query, (
            user_id, filename, file_hash, file_type,
            page_count, has_tables, has_charts, size_bytes
        ))
        return cur.fetchone()

    def get_user_documents(self, user_id: str):
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
            SELECT doc_id, filename, uploaded_at,
                   page_count, has_tables, has_charts, size_bytes, summary
            FROM documents
            WHERE user_id=%s
            ORDER BY uploaded_at DESC
        """, (user_id,))
        return cur.fetchall()

    def get_document_summary(self, doc_id: str) -> str:
        cur = self.conn.cursor()
        cur.execute("SELECT summary FROM documents WHERE doc_id=%s", (doc_id,))
        row = cur.fetchone()
        return row[0] if row else ""

    def update_summary(self, doc_id: str, summary: str):
        cur = self.conn.cursor()
        cur.execute("UPDATE documents SET summary=%s WHERE doc_id=%s", (summary, doc_id))

    def delete_document(self, doc_id: str):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM documents WHERE doc_id=%s", (doc_id,))

    def get_document_metadata(self, doc_id: str):
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT * FROM documents WHERE doc_id=%s", (doc_id,))
        return cur.fetchone()
