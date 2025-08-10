import os
import sys
import unittest
from unittest.mock import patch

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem


class MockConfig:
    """Mock configuration for testing"""

    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    CHROMA_PATH = "./test_chroma_db"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MAX_RESULTS = 5
    ANTHROPIC_API_KEY = "test_api_key"
    ANTHROPIC_MODEL = "claude-3-sonnet"
    MAX_HISTORY = 2


class TestRAGSystem(unittest.TestCase):
    """Test suite for RAG system content query handling"""

    def setUp(self):
        """Set up test fixtures"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
        ):
            self.mock_config = MockConfig()
            self.rag_system = RAGSystem(self.mock_config)

            # Set up mocks for easier access
            self.mock_document_processor = self.rag_system.document_processor
            self.mock_vector_store = self.rag_system.vector_store
            self.mock_ai_generator = self.rag_system.ai_generator
            self.mock_session_manager = self.rag_system.session_manager
            self.mock_tool_manager = self.rag_system.tool_manager

    def test_rag_system_initialization(self):
        """Test RAG system initializes all components correctly"""
        # Verify all components are initialized
        self.assertIsNotNone(self.rag_system.document_processor)
        self.assertIsNotNone(self.rag_system.vector_store)
        self.assertIsNotNone(self.rag_system.ai_generator)
        self.assertIsNotNone(self.rag_system.session_manager)
        self.assertIsNotNone(self.rag_system.tool_manager)
        self.assertIsNotNone(self.rag_system.search_tool)
        self.assertIsNotNone(self.rag_system.outline_tool)

    def test_query_without_session_success(self):
        """Test successful query processing without session"""
        # Mock AI response and sources
        mock_response = "Machine learning is a subset of artificial intelligence..."
        mock_sources = [
            {
                "text": "AI Fundamentals - Lesson 1",
                "link": "https://example.com/lesson1",
            },
            {"text": "ML Basics - Lesson 2", "link": None},
        ]

        self.mock_ai_generator.generate_response.return_value = mock_response
        self.mock_tool_manager.get_last_sources.return_value = mock_sources

        # Execute query
        response, sources = self.rag_system.query("What is machine learning?")

        # Verify AI generator was called correctly
        self.mock_ai_generator.generate_response.assert_called_once()
        call_args = self.mock_ai_generator.generate_response.call_args

        # Check the prompt format
        expected_prompt = (
            "Answer this question about course materials: What is machine learning?"
        )
        self.assertEqual(call_args[1]["query"], expected_prompt)

        # Check tools were provided
        self.assertIsNotNone(call_args[1]["tools"])
        self.assertEqual(call_args[1]["tool_manager"], self.mock_tool_manager)

        # Verify response and sources
        self.assertEqual(response, mock_response)
        self.assertEqual(sources, mock_sources)

        # Verify sources were retrieved and reset
        self.mock_tool_manager.get_last_sources.assert_called_once()
        self.mock_tool_manager.reset_sources.assert_called_once()

    def test_query_with_session_success(self):
        """Test successful query processing with session management"""
        # Mock session data
        session_id = "test_session_123"
        mock_history = "User: What is AI?\nAI: AI is artificial intelligence."
        mock_response = "Deep learning is a subset of machine learning..."

        self.mock_session_manager.get_conversation_history.return_value = mock_history
        self.mock_ai_generator.generate_response.return_value = mock_response
        self.mock_tool_manager.get_last_sources.return_value = []

        # Execute query with session
        response, sources = self.rag_system.query("What is deep learning?", session_id)

        # Verify session history was retrieved
        self.mock_session_manager.get_conversation_history.assert_called_once_with(
            session_id
        )

        # Verify AI generator got conversation history
        call_args = self.mock_ai_generator.generate_response.call_args
        self.assertEqual(call_args[1]["conversation_history"], mock_history)

        # Verify session was updated
        self.mock_session_manager.add_exchange.assert_called_once_with(
            session_id, "What is deep learning?", mock_response
        )

    def test_query_with_content_specific_question(self):
        """Test query that should trigger course search tool"""
        # Mock course-specific response
        mock_response = "According to the MCP course materials..."
        mock_sources = [
            {
                "text": "MCP: Build Rich-Context AI Apps - Lesson 1",
                "link": "https://example.com/mcp1",
            }
        ]

        self.mock_ai_generator.generate_response.return_value = mock_response
        self.mock_tool_manager.get_last_sources.return_value = mock_sources

        # Execute course-specific query
        response, sources = self.rag_system.query("How do I implement MCP servers?")

        # Verify the prompt mentions course materials
        call_args = self.mock_ai_generator.generate_response.call_args
        expected_prompt = "Answer this question about course materials: How do I implement MCP servers?"
        self.assertEqual(call_args[1]["query"], expected_prompt)

        # Verify tools are available for course search
        self.assertIsNotNone(call_args[1]["tools"])

        # Verify response includes course content
        self.assertEqual(response, mock_response)
        self.assertEqual(len(sources), 1)
        self.assertIn("MCP", sources[0]["text"])

    def test_query_general_knowledge_question(self):
        """Test query that doesn't require course search"""
        # Mock general response without sources
        mock_response = "The capital of France is Paris."

        self.mock_ai_generator.generate_response.return_value = mock_response
        self.mock_tool_manager.get_last_sources.return_value = []

        # Execute general query
        response, sources = self.rag_system.query("What is the capital of France?")

        # Verify response
        self.assertEqual(response, mock_response)
        self.assertEqual(len(sources), 0)

        # Verify AI still gets tools (it decides whether to use them)
        call_args = self.mock_ai_generator.generate_response.call_args
        self.assertIsNotNone(call_args[1]["tools"])

    def test_query_error_handling(self):
        """Test error handling during query processing"""
        # Mock AI generator to raise exception
        self.mock_ai_generator.generate_response.side_effect = Exception("API Error")

        # Execute query and expect exception to propagate
        with self.assertRaises(Exception) as context:
            self.rag_system.query("Test query")

        self.assertIn("API Error", str(context.exception))

    def test_add_course_document_success(self):
        """Test successful addition of a single course document"""
        # Mock course and chunks
        mock_course = Course(
            title="Test Course",
            instructor="Test Instructor",
            course_link="https://example.com/course",
            lessons=[Lesson(lesson_number=1, title="Lesson 1")],
        )

        mock_chunks = [
            CourseChunk(
                content="Test content chunk 1",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0,
            ),
            CourseChunk(
                content="Test content chunk 2",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=1,
            ),
        ]

        # Mock document processor
        self.mock_document_processor.process_course_document.return_value = (
            mock_course,
            mock_chunks,
        )

        # Add course document
        course, chunk_count = self.rag_system.add_course_document("/path/to/course.txt")

        # Verify processing
        self.mock_document_processor.process_course_document.assert_called_once_with(
            "/path/to/course.txt"
        )

        # Verify vector store operations
        self.mock_vector_store.add_course_metadata.assert_called_once_with(mock_course)
        self.mock_vector_store.add_course_content.assert_called_once_with(mock_chunks)

        # Verify return values
        self.assertEqual(course.title, "Test Course")
        self.assertEqual(chunk_count, 2)

    def test_add_course_document_error(self):
        """Test error handling when adding course document fails"""
        # Mock document processor to raise exception
        self.mock_document_processor.process_course_document.side_effect = Exception(
            "Processing failed"
        )

        # Add course document
        course, chunk_count = self.rag_system.add_course_document(
            "/path/to/bad_file.txt"
        )

        # Verify error handling
        self.assertIsNone(course)
        self.assertEqual(chunk_count, 0)

    @patch("os.path.exists")
    @patch("os.listdir")
    def test_add_course_folder_success(self, mock_listdir, mock_exists):
        """Test successful addition of course folder"""
        # Mock file system
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt", "course2.pdf", "readme.md"]

        # Mock existing courses
        self.mock_vector_store.get_existing_course_titles.return_value = [
            "Existing Course"
        ]

        # Mock document processing
        mock_course1 = Course(title="New Course 1", lessons=[])
        mock_course2 = Course(title="New Course 2", lessons=[])
        mock_chunks1 = [
            CourseChunk(content="Content 1", course_title="New Course 1", chunk_index=0)
        ]
        mock_chunks2 = [
            CourseChunk(content="Content 2", course_title="New Course 2", chunk_index=0)
        ]

        self.mock_document_processor.process_course_document.side_effect = [
            (mock_course1, mock_chunks1),
            (mock_course2, mock_chunks2),
        ]

        # Add course folder
        total_courses, total_chunks = self.rag_system.add_course_folder(
            "/path/to/courses"
        )

        # Verify results
        self.assertEqual(total_courses, 2)
        self.assertEqual(total_chunks, 2)

        # Verify only valid files were processed
        self.assertEqual(
            self.mock_document_processor.process_course_document.call_count, 2
        )

    @patch("os.path.exists")
    def test_add_course_folder_missing_path(self, mock_exists):
        """Test handling of non-existent folder path"""
        mock_exists.return_value = False

        # Add non-existent folder
        total_courses, total_chunks = self.rag_system.add_course_folder(
            "/path/to/missing"
        )

        # Verify no processing occurred
        self.assertEqual(total_courses, 0)
        self.assertEqual(total_chunks, 0)
        self.mock_document_processor.process_course_document.assert_not_called()

    @patch("os.path.exists")
    @patch("os.listdir")
    def test_add_course_folder_skip_existing(self, mock_listdir, mock_exists):
        """Test skipping of existing courses"""
        # Mock file system
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt"]

        # Mock existing course
        existing_title = "Existing Course"
        self.mock_vector_store.get_existing_course_titles.return_value = [
            existing_title
        ]

        # Mock document processing to return existing course
        mock_course = Course(title=existing_title, lessons=[])
        mock_chunks = []
        self.mock_document_processor.process_course_document.return_value = (
            mock_course,
            mock_chunks,
        )

        # Add course folder
        total_courses, total_chunks = self.rag_system.add_course_folder(
            "/path/to/courses"
        )

        # Verify existing course was skipped
        self.assertEqual(total_courses, 0)
        self.assertEqual(total_chunks, 0)
        self.mock_vector_store.add_course_metadata.assert_not_called()
        self.mock_vector_store.add_course_content.assert_not_called()

    def test_get_course_analytics(self):
        """Test course analytics retrieval"""
        # Mock vector store analytics
        mock_course_count = 4
        mock_course_titles = ["Course 1", "Course 2", "Course 3", "Course 4"]

        self.mock_vector_store.get_course_count.return_value = mock_course_count
        self.mock_vector_store.get_existing_course_titles.return_value = (
            mock_course_titles
        )

        # Get analytics
        analytics = self.rag_system.get_course_analytics()

        # Verify analytics
        self.assertEqual(analytics["total_courses"], mock_course_count)
        self.assertEqual(analytics["course_titles"], mock_course_titles)

    def test_query_prompt_format(self):
        """Test that query prompts are formatted correctly"""
        # Mock response
        self.mock_ai_generator.generate_response.return_value = "Test response"
        self.mock_tool_manager.get_last_sources.return_value = []

        # Test different query types
        test_queries = [
            "What is machine learning?",
            "How do I use MCP servers?",
            "Explain neural networks",
            "",  # Empty query
        ]

        for query in test_queries:
            with self.subTest(query=query):
                self.rag_system.query(query)

                # Verify prompt format
                call_args = self.mock_ai_generator.generate_response.call_args
                expected_prompt = (
                    f"Answer this question about course materials: {query}"
                )
                self.assertEqual(call_args[1]["query"], expected_prompt)

    def test_tool_registration(self):
        """Test that tools are properly registered"""
        # Verify tools were registered in tool manager
        # Note: This requires checking the actual registration in setup
        # Since we're using mocks, we'll verify the setup calls

        # The setup should have called register_tool twice
        self.assertIsNotNone(self.rag_system.search_tool)
        self.assertIsNotNone(self.rag_system.outline_tool)


if __name__ == "__main__":
    unittest.main()
