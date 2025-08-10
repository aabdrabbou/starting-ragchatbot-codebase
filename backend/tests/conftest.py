"""
Shared test fixtures and configuration for RAG system tests
"""

import pytest
import tempfile
import shutil
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from rag_system import RAGSystem
from vector_store import VectorStore
from document_processor import DocumentProcessor
from ai_generator import AIGenerator
from session_manager import SessionManager


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration with temporary paths"""
    config = Config()
    config.chroma_db_path = os.path.join(temp_dir, "test_chroma")
    config.chunk_size = 100  # Smaller for faster tests
    config.chunk_overlap = 20
    return config


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing AI generation"""
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="Test response from AI")]
    mock_client.messages.create.return_value = mock_message
    return mock_client


@pytest.fixture
def test_courses_data():
    """Sample course data for testing"""
    return [
        {
            "title": "Introduction to Python",
            "instructor": "Jane Doe",
            "lessons": [
                {"number": 1, "title": "Variables and Data Types", "content": "Learn about Python variables and basic data types like strings, integers, and floats."},
                {"number": 2, "title": "Control Flow", "content": "Understand if statements, loops, and how to control program execution flow."}
            ]
        },
        {
            "title": "Advanced JavaScript",
            "instructor": "John Smith", 
            "lessons": [
                {"number": 1, "title": "Async Programming", "content": "Master promises, async/await, and asynchronous JavaScript programming patterns."},
                {"number": 2, "title": "Module Systems", "content": "Learn ES6 modules, CommonJS, and different ways to organize JavaScript code."}
            ]
        }
    ]


@pytest.fixture
def mock_vector_store(test_config):
    """Mock vector store for testing"""
    mock_store = MagicMock(spec=VectorStore)
    mock_store.get_collection.return_value = MagicMock()
    mock_store.query_content.return_value = [
        {
            "content": "Test content about Python variables",
            "course_title": "Introduction to Python",
            "lesson_title": "Variables and Data Types"
        }
    ]
    mock_store.query_courses.return_value = [
        {"title": "Introduction to Python", "instructor": "Jane Doe"}
    ]
    return mock_store


@pytest.fixture
def mock_document_processor():
    """Mock document processor for testing"""
    mock_processor = MagicMock(spec=DocumentProcessor)
    mock_processor.process_course_folder.return_value = [
        MagicMock(title="Test Course", instructor="Test Instructor")
    ]
    return mock_processor


@pytest.fixture
def mock_ai_generator():
    """Mock AI generator for testing"""
    mock_generator = MagicMock(spec=AIGenerator)
    mock_generator.generate_response.return_value = "Mock AI response with context"
    return mock_generator


@pytest.fixture
def mock_session_manager():
    """Mock session manager for testing"""
    mock_manager = MagicMock(spec=SessionManager)
    mock_manager.create_session.return_value = "test-session-123"
    mock_manager.get_session_history.return_value = []
    mock_manager.add_to_session.return_value = None
    return mock_manager


@pytest.fixture
def test_rag_system(test_config, mock_vector_store, mock_document_processor, 
                   mock_ai_generator, mock_session_manager):
    """Create a RAG system with mocked components for testing"""
    rag_system = RAGSystem(test_config)
    rag_system.vector_store = mock_vector_store
    rag_system.document_processor = mock_document_processor
    rag_system.ai_generator = mock_ai_generator
    rag_system.session_manager = mock_session_manager
    return rag_system


@pytest.fixture
def sample_query_request():
    """Sample query request data for API testing"""
    return {
        "query": "What are Python data types?",
        "session_id": "test-session-123"
    }


@pytest.fixture
def sample_query_response():
    """Sample query response data for API testing"""
    return {
        "answer": "Python has several basic data types including strings, integers, floats, and booleans.",
        "sources": [
            {"text": "Learn about Python variables and basic data types", "link": None}
        ],
        "session_id": "test-session-123"
    }


@pytest.fixture
def sample_course_stats():
    """Sample course statistics for API testing"""
    return {
        "total_courses": 2,
        "course_titles": ["Introduction to Python", "Advanced JavaScript"]
    }


@pytest.fixture
def test_documents_dir(temp_dir, test_courses_data):
    """Create a temporary documents directory with test files"""
    docs_dir = os.path.join(temp_dir, "docs")
    os.makedirs(docs_dir)
    
    # Create test course files
    for i, course in enumerate(test_courses_data):
        course_file = os.path.join(docs_dir, f"course_{i+1}.txt")
        content = f"Title: {course['title']}\nInstructor: {course['instructor']}\n\n"
        for lesson in course['lessons']:
            content += f"Lesson {lesson['number']}: {lesson['title']}\n{lesson['content']}\n\n"
        
        with open(course_file, 'w') as f:
            f.write(content)
    
    return docs_dir


@pytest.fixture(autouse=True)
def setup_env_vars(monkeypatch):
    """Set up environment variables for testing"""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")


@pytest.fixture
def mock_startup_event(monkeypatch):
    """Mock the startup event to prevent actual document loading during API tests"""
    async def mock_startup():
        pass
    
    monkeypatch.setattr("app.startup_event", mock_startup)


@pytest.fixture
def mock_static_files(monkeypatch):
    """Mock static files mounting to prevent filesystem issues during API tests"""
    def mock_mount(*args, **kwargs):
        pass
    
    monkeypatch.setattr("fastapi.FastAPI.mount", mock_mount)