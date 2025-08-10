import os
import sys
import unittest
from unittest.mock import Mock

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from search_tools import CourseSearchTool
from vector_store import SearchResults, VectorStore


class TestCourseSearchTool(unittest.TestCase):
    """Test suite for CourseSearchTool execute method"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock vector store
        self.mock_vector_store = Mock(spec=VectorStore)
        self.search_tool = CourseSearchTool(self.mock_vector_store)

    def test_execute_successful_search_with_results(self):
        """Test successful search that returns results"""
        # Mock search results
        mock_results = SearchResults(
            documents=["Test content about machine learning", "Another relevant chunk"],
            metadata=[
                {"course_title": "AI Fundamentals", "lesson_number": 1},
                {"course_title": "AI Fundamentals", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
            error=None,
        )

        # Mock get_lesson_link to return None (no links)
        self.mock_vector_store.get_lesson_link.return_value = None
        self.mock_vector_store.search.return_value = mock_results

        # Execute search
        result = self.search_tool.execute("machine learning")

        # Verify search was called correctly
        self.mock_vector_store.search.assert_called_once_with(
            query="machine learning", course_name=None, lesson_number=None
        )

        # Verify result format
        self.assertIn("[AI Fundamentals - Lesson 1]", result)
        self.assertIn("[AI Fundamentals - Lesson 2]", result)
        self.assertIn("Test content about machine learning", result)
        self.assertIn("Another relevant chunk", result)

        # Verify sources were tracked
        self.assertEqual(len(self.search_tool.last_sources), 2)
        self.assertEqual(
            self.search_tool.last_sources[0]["text"], "AI Fundamentals - Lesson 1"
        )
        self.assertEqual(
            self.search_tool.last_sources[1]["text"], "AI Fundamentals - Lesson 2"
        )

    def test_execute_with_course_name_filter(self):
        """Test search with course name filter"""
        mock_results = SearchResults(
            documents=["MCP specific content"],
            metadata=[
                {"course_title": "MCP: Build Rich-Context AI Apps", "lesson_number": 3}
            ],
            distances=[0.15],
            error=None,
        )

        self.mock_vector_store.get_lesson_link.return_value = (
            "https://example.com/lesson3"
        )
        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute("context", course_name="MCP")

        # Verify search parameters
        self.mock_vector_store.search.assert_called_once_with(
            query="context", course_name="MCP", lesson_number=None
        )

        # Verify lesson link was retrieved
        self.mock_vector_store.get_lesson_link.assert_called_once_with(
            "MCP: Build Rich-Context AI Apps", 3
        )

        # Verify result format and source with link
        self.assertIn("[MCP: Build Rich-Context AI Apps - Lesson 3]", result)
        self.assertEqual(
            self.search_tool.last_sources[0]["link"], "https://example.com/lesson3"
        )

    def test_execute_with_lesson_number_filter(self):
        """Test search with lesson number filter"""
        mock_results = SearchResults(
            documents=["Lesson 1 specific content"],
            metadata=[{"course_title": "Introduction to AI", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )

        self.mock_vector_store.get_lesson_link.return_value = None
        self.mock_vector_store.search.return_value = mock_results

        _result = self.search_tool.execute("introduction", lesson_number=1)

        # Verify search parameters
        self.mock_vector_store.search.assert_called_once_with(
            query="introduction", course_name=None, lesson_number=1
        )

    def test_execute_with_both_filters(self):
        """Test search with both course name and lesson number filters"""
        mock_results = SearchResults(
            documents=["Specific lesson content"],
            metadata=[{"course_title": "Advanced AI", "lesson_number": 2}],
            distances=[0.05],
            error=None,
        )

        self.mock_vector_store.get_lesson_link.return_value = None
        self.mock_vector_store.search.return_value = mock_results

        _result = self.search_tool.execute(
            "advanced", course_name="Advanced AI", lesson_number=2
        )

        # Verify search parameters
        self.mock_vector_store.search.assert_called_once_with(
            query="advanced", course_name="Advanced AI", lesson_number=2
        )

    def test_execute_search_error(self):
        """Test handling of search errors"""
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error="Database connection failed"
        )

        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute("test query")

        # Verify error is returned
        self.assertEqual(result, "Database connection failed")

        # Verify no sources are tracked on error
        self.assertEqual(len(self.search_tool.last_sources), 0)

    def test_execute_no_results_found(self):
        """Test handling when no results are found"""
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )

        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute("nonexistent topic")

        # Verify appropriate message
        self.assertEqual(result, "No relevant content found.")

    def test_execute_no_results_with_course_filter(self):
        """Test no results message with course filter"""
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )

        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute("test", course_name="Missing Course")

        # Verify filter is mentioned in message
        self.assertEqual(
            result, "No relevant content found in course 'Missing Course'."
        )

    def test_execute_no_results_with_lesson_filter(self):
        """Test no results message with lesson filter"""
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )

        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute("test", lesson_number=5)

        # Verify filter is mentioned in message
        self.assertEqual(result, "No relevant content found in lesson 5.")

    def test_execute_no_results_with_both_filters(self):
        """Test no results message with both filters"""
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )

        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute(
            "test", course_name="Test Course", lesson_number=3
        )

        # Verify both filters are mentioned in message
        self.assertEqual(
            result, "No relevant content found in course 'Test Course' in lesson 3."
        )

    def test_execute_missing_metadata(self):
        """Test handling when metadata is incomplete"""
        mock_results = SearchResults(
            documents=["Content without proper metadata"],
            metadata=[{}],  # Empty metadata
            distances=[0.1],
            error=None,
        )

        self.mock_vector_store.get_lesson_link.return_value = None
        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute("test")

        # Verify it handles missing metadata gracefully
        self.assertIn("[unknown]", result)
        self.assertIn("Content without proper metadata", result)

    def test_get_tool_definition(self):
        """Test tool definition for Anthropic integration"""
        definition = self.search_tool.get_tool_definition()

        # Verify required fields
        self.assertEqual(definition["name"], "search_course_content")
        self.assertIn("description", definition)
        self.assertIn("input_schema", definition)

        # Verify schema structure
        schema = definition["input_schema"]
        self.assertEqual(schema["type"], "object")
        self.assertIn("query", schema["properties"])
        self.assertIn("course_name", schema["properties"])
        self.assertIn("lesson_number", schema["properties"])
        self.assertEqual(schema["required"], ["query"])

    def test_format_results_with_links(self):
        """Test result formatting when lesson links are available"""
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )

        # Mock lesson link
        self.mock_vector_store.get_lesson_link.return_value = (
            "https://example.com/lesson1"
        )

        formatted = self.search_tool._format_results(mock_results)

        # Verify formatted output
        self.assertIn("[Test Course - Lesson 1]", formatted)
        self.assertIn("Test content", formatted)

        # Verify source with link
        self.assertEqual(len(self.search_tool.last_sources), 1)
        self.assertEqual(
            self.search_tool.last_sources[0]["text"], "Test Course - Lesson 1"
        )
        self.assertEqual(
            self.search_tool.last_sources[0]["link"], "https://example.com/lesson1"
        )


if __name__ == "__main__":
    unittest.main()
