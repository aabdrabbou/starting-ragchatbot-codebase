import os
import shutil
import sys
import tempfile
import unittest

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Course, CourseChunk, Lesson
from vector_store import SearchResults, VectorStore


class TestVectorStoreIntegration(unittest.TestCase):
    """Integration tests for VectorStore with real ChromaDB operations"""

    def setUp(self):
        """Set up test fixtures with temporary database"""
        # Create temporary directory for test database
        self.test_db_dir = tempfile.mkdtemp()

        # Initialize vector store with test database
        self.vector_store = VectorStore(
            chroma_path=self.test_db_dir,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,
        )

    def tearDown(self):
        """Clean up test database"""
        try:
            shutil.rmtree(self.test_db_dir)
        except Exception as e:
            print(f"Warning: Could not clean up test database: {e}")

    def test_vector_store_initialization(self):
        """Test vector store initializes correctly"""
        # Verify collections exist
        self.assertIsNotNone(self.vector_store.course_catalog)
        self.assertIsNotNone(self.vector_store.course_content)

        # Verify initial state is empty
        self.assertEqual(self.vector_store.get_course_count(), 0)
        self.assertEqual(len(self.vector_store.get_existing_course_titles()), 0)

    def test_add_course_metadata(self):
        """Test adding course metadata to catalog"""
        # Create test course
        test_course = Course(
            title="Test AI Course",
            instructor="Test Instructor",
            course_link="https://example.com/course",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Introduction",
                    lesson_link="https://example.com/lesson1",
                ),
                Lesson(
                    lesson_number=2,
                    title="Advanced Topics",
                    lesson_link="https://example.com/lesson2",
                ),
            ],
        )

        # Add to vector store
        self.vector_store.add_course_metadata(test_course)

        # Verify course was added
        self.assertEqual(self.vector_store.get_course_count(), 1)

        # Verify course titles
        titles = self.vector_store.get_existing_course_titles()
        self.assertEqual(len(titles), 1)
        self.assertEqual(titles[0], "Test AI Course")

        # Verify metadata retrieval
        metadata_list = self.vector_store.get_all_courses_metadata()
        self.assertEqual(len(metadata_list), 1)

        metadata = metadata_list[0]
        self.assertEqual(metadata["title"], "Test AI Course")
        self.assertEqual(metadata["instructor"], "Test Instructor")
        self.assertEqual(metadata["course_link"], "https://example.com/course")
        self.assertEqual(len(metadata["lessons"]), 2)

    def test_add_course_content(self):
        """Test adding course content chunks"""
        # Create test chunks
        test_chunks = [
            CourseChunk(
                content="This is the introduction to artificial intelligence and machine learning concepts.",
                course_title="AI Fundamentals",
                lesson_number=1,
                chunk_index=0,
            ),
            CourseChunk(
                content="Machine learning algorithms can be supervised, unsupervised, or reinforcement learning.",
                course_title="AI Fundamentals",
                lesson_number=1,
                chunk_index=1,
            ),
            CourseChunk(
                content="Neural networks are inspired by biological neurons and form the basis of deep learning.",
                course_title="AI Fundamentals",
                lesson_number=2,
                chunk_index=2,
            ),
        ]

        # Add chunks to vector store
        self.vector_store.add_course_content(test_chunks)

        # Test basic search
        results = self.vector_store.search("machine learning")

        # Verify search works
        self.assertFalse(results.is_empty())
        self.assertIsNone(results.error)
        self.assertGreater(len(results.documents), 0)

        # Verify metadata is preserved
        self.assertEqual(len(results.metadata), len(results.documents))
        for metadata in results.metadata:
            self.assertEqual(metadata["course_title"], "AI Fundamentals")
            self.assertIn("lesson_number", metadata)
            self.assertIn("chunk_index", metadata)

    def test_search_with_course_filter(self):
        """Test search with course name filtering"""
        # Add content from multiple courses
        course1_chunks = [
            CourseChunk(
                content="Python programming basics",
                course_title="Python Course",
                lesson_number=1,
                chunk_index=0,
            ),
            CourseChunk(
                content="Advanced Python concepts",
                course_title="Python Course",
                lesson_number=2,
                chunk_index=1,
            ),
        ]

        course2_chunks = [
            CourseChunk(
                content="JavaScript fundamentals",
                course_title="JavaScript Course",
                lesson_number=1,
                chunk_index=0,
            ),
            CourseChunk(
                content="React and modern JavaScript",
                course_title="JavaScript Course",
                lesson_number=2,
                chunk_index=1,
            ),
        ]

        # Add courses to catalog first
        python_course = Course(title="Python Course", lessons=[])
        js_course = Course(title="JavaScript Course", lessons=[])

        self.vector_store.add_course_metadata(python_course)
        self.vector_store.add_course_metadata(js_course)

        # Add content
        self.vector_store.add_course_content(course1_chunks + course2_chunks)

        # Search with course filter
        results = self.vector_store.search("programming", course_name="Python")

        # Verify filtered results
        self.assertFalse(results.is_empty())
        for metadata in results.metadata:
            self.assertEqual(metadata["course_title"], "Python Course")

    def test_search_with_lesson_filter(self):
        """Test search with lesson number filtering"""
        # Add content with multiple lessons
        test_chunks = [
            CourseChunk(
                content="Lesson 1 content about basics",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0,
            ),
            CourseChunk(
                content="Lesson 2 content about advanced topics",
                course_title="Test Course",
                lesson_number=2,
                chunk_index=1,
            ),
            CourseChunk(
                content="Lesson 3 content about expert level",
                course_title="Test Course",
                lesson_number=3,
                chunk_index=2,
            ),
        ]

        self.vector_store.add_course_content(test_chunks)

        # Search with lesson filter
        results = self.vector_store.search("content", lesson_number=2)

        # Verify filtered results
        self.assertFalse(results.is_empty())
        for metadata in results.metadata:
            self.assertEqual(metadata["lesson_number"], 2)

    def test_search_no_results(self):
        """Test search when no results are found"""
        # Add some content
        test_chunks = [
            CourseChunk(
                content="Python programming",
                course_title="Python Course",
                lesson_number=1,
                chunk_index=0,
            )
        ]
        self.vector_store.add_course_content(test_chunks)

        # Search for non-existent content
        results = self.vector_store.search("quantum computing")

        # Verify empty results
        self.assertTrue(results.is_empty())
        self.assertIsNone(results.error)
        self.assertEqual(len(results.documents), 0)

    def test_course_name_resolution(self):
        """Test course name resolution with partial matches"""
        # Add courses to catalog
        courses = [
            Course(title="Machine Learning Fundamentals", lessons=[]),
            Course(title="Deep Learning Advanced", lessons=[]),
            Course(title="MCP: Build Rich-Context AI Apps", lessons=[]),
        ]

        for course in courses:
            self.vector_store.add_course_metadata(course)

        # Test exact match
        resolved = self.vector_store._resolve_course_name(
            "Machine Learning Fundamentals"
        )
        self.assertEqual(resolved, "Machine Learning Fundamentals")

        # Test partial match (this will depend on semantic similarity)
        resolved = self.vector_store._resolve_course_name("MCP")
        # Should match "MCP: Build Rich-Context AI Apps"
        self.assertIsNotNone(resolved)

    def test_get_lesson_link(self):
        """Test lesson link retrieval"""
        # Create course with lessons
        test_course = Course(
            title="Test Course",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Intro",
                    lesson_link="https://example.com/lesson1",
                ),
                Lesson(
                    lesson_number=2,
                    title="Advanced",
                    lesson_link="https://example.com/lesson2",
                ),
            ],
        )

        self.vector_store.add_course_metadata(test_course)

        # Test lesson link retrieval
        link1 = self.vector_store.get_lesson_link("Test Course", 1)
        self.assertEqual(link1, "https://example.com/lesson1")

        link2 = self.vector_store.get_lesson_link("Test Course", 2)
        self.assertEqual(link2, "https://example.com/lesson2")

        # Test non-existent lesson
        link3 = self.vector_store.get_lesson_link("Test Course", 99)
        self.assertIsNone(link3)

    def test_clear_all_data(self):
        """Test clearing all data from vector store"""
        # Add some data
        test_course = Course(title="Test Course", lessons=[])
        test_chunks = [
            CourseChunk(
                content="Test content",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0,
            )
        ]

        self.vector_store.add_course_metadata(test_course)
        self.vector_store.add_course_content(test_chunks)

        # Verify data exists
        self.assertEqual(self.vector_store.get_course_count(), 1)

        # Clear all data
        self.vector_store.clear_all_data()

        # Verify data is cleared
        self.assertEqual(self.vector_store.get_course_count(), 0)
        self.assertEqual(len(self.vector_store.get_existing_course_titles()), 0)

        # Verify search returns empty
        results = self.vector_store.search("test")
        self.assertTrue(results.is_empty())

    def test_search_results_structure(self):
        """Test SearchResults structure and methods"""
        # Create empty results
        empty_results = SearchResults.empty("No results found")
        self.assertTrue(empty_results.is_empty())
        self.assertEqual(empty_results.error, "No results found")

        # Create results with data
        data_results = SearchResults(
            documents=["doc1", "doc2"],
            metadata=[{"key": "value1"}, {"key": "value2"}],
            distances=[0.1, 0.2],
        )
        self.assertFalse(data_results.is_empty())
        self.assertIsNone(data_results.error)
        self.assertEqual(len(data_results.documents), 2)

    def test_empty_chunks_handling(self):
        """Test handling of empty chunks list"""
        # Try adding empty chunks list
        self.vector_store.add_course_content([])

        # Should handle gracefully without errors
        results = self.vector_store.search("anything")
        self.assertTrue(results.is_empty())


if __name__ == "__main__":
    unittest.main()
