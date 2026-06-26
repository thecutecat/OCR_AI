import pyodbc
from typing import Dict, Any, Optional
import logging
import json
from datetime import datetime

class DatabaseManager:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """
        Initialize database tables if they don't exist.
        """
        try:
            conn = pyodbc.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Create documents table
            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='documents' AND xtype='U')
                CREATE TABLE documents (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    document_id VARCHAR(50) UNIQUE NOT NULL,
                    document_type VARCHAR(50),
                    image_path VARCHAR(500),
                    processed_image_path VARCHAR(500),
                    metadata NVARCHAR(MAX),
                    ocr_result NVARCHAR(MAX),
                    created_at DATETIME DEFAULT GETDATE()
                )
            """)
            
            # Create processing_logs table
            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='processing_logs' AND xtype='U')
                CREATE TABLE processing_logs (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    document_id VARCHAR(50),
                    processing_time_ms FLOAT,
                    ocr_confidence FLOAT,
                    success BIT,
                    error_message NVARCHAR(MAX),
                    created_at DATETIME DEFAULT GETDATE()
                )
            """)
            
            conn.commit()
            conn.close()
            self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
    
    def save_document(
        self,
        document_id: str,
        document_type: str,
        image_path: str,
        processed_image_path: str,
        metadata: Dict[str, Any],
        ocr_result: Dict[str, Any]
    ) -> bool:
        """
        Save document processing results to database.
        """
        try:
            conn = pyodbc.connect(self.connection_string)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO documents 
                (document_id, document_type, image_path, processed_image_path, metadata, ocr_result)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                document_id,
                document_type,
                image_path,
                processed_image_path,
                json.dumps(metadata),
                json.dumps(ocr_result)
            ))
            
            conn.commit()
            conn.close()
            self.logger.info(f"Document {document_id} saved to database")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save document: {e}")
            return False
    
    def save_processing_log(
        self,
        document_id: str,
        processing_time_ms: float,
        ocr_confidence: float,
        success: bool,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Save processing log to database.
        """
        try:
            conn = pyodbc.connect(self.connection_string)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO processing_logs 
                (document_id, processing_time_ms, ocr_confidence, success, error_message)
                VALUES (?, ?, ?, ?, ?)
            """, (
                document_id,
                processing_time_ms,
                ocr_confidence,
                success,
                error_message
            ))
            
            conn.commit()
            conn.close()
            self.logger.info(f"Processing log for {document_id} saved")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save processing log: {e}")
            return False
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve document data from database.
        """
        try:
            conn = pyodbc.connect(self.connection_string)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM documents WHERE document_id = ?
            """, (document_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'id': row[0],
                    'document_id': row[1],
                    'document_type': row[2],
                    'image_path': row[3],
                    'processed_image_path': row[4],
                    'metadata': json.loads(row[5]),
                    'ocr_result': json.loads(row[6]),
                    'created_at': row[7]
                }
            
            return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve document: {e}")
            return None