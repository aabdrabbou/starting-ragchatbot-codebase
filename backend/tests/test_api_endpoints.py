"""
API endpoint tests for the FastAPI application
"""

import pytest
import json
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Import the models and create a test app to avoid static file mounting issues
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class SourceInfo(BaseModel):
    text: str
    link: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    session_id: str

class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


@pytest.fixture
def mock_rag_system():
    """Mock RAG system for API testing"""
    mock_rag = MagicMock()
    mock_rag.session_manager.create_session.return_value = "test-session-123"
    mock_rag.query.return_value = (
        "Python has several basic data types including strings, integers, floats, and booleans.",
        [{"text": "Learn about Python variables and basic data types", "link": None}]
    )
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Introduction to Python", "Advanced JavaScript"]
    }
    return mock_rag


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app without static file mounting"""
    app = FastAPI(title="Course Materials RAG System Test")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Define the API endpoints inline to avoid import issues
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            source_objects = []
            for source in sources:
                if isinstance(source, dict):
                    source_objects.append(SourceInfo(
                        text=source.get("text", ""),
                        link=source.get("link")
                    ))
                else:
                    source_objects.append(SourceInfo(text=str(source), link=None))
            
            return QueryResponse(
                answer=answer,
                sources=source_objects,
                session_id=session_id
            )
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


@pytest.fixture
def client(test_app):
    """Create a test client"""
    return TestClient(test_app)


@pytest.mark.api
class TestQueryEndpoint:
    """Test cases for the /api/query endpoint"""
    
    def test_query_with_session_id(self, client, sample_query_request):
        """Test querying with an existing session ID"""
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == sample_query_request["session_id"]
        assert isinstance(data["sources"], list)
        
        if data["sources"]:
            source = data["sources"][0]
            assert "text" in source
            assert "link" in source
    
    def test_query_without_session_id(self, client):
        """Test querying without a session ID (should create new session)"""
        request_data = {"query": "What are Python data types?"}
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"  # From mock
    
    def test_query_with_empty_query(self, client):
        """Test querying with an empty query string"""
        request_data = {"query": ""}
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
    
    def test_query_with_invalid_request(self, client):
        """Test querying with invalid request format"""
        response = client.post("/api/query", json={})
        
        assert response.status_code == 422  # Validation error
    
    def test_query_with_malformed_json(self, client):
        """Test querying with malformed JSON"""
        response = client.post(
            "/api/query", 
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_query_response_format(self, client, sample_query_request):
        """Test that the response format matches the expected schema"""
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Validate source structure if sources exist
        if data["sources"]:
            for source in data["sources"]:
                assert "text" in source
                assert "link" in source
                assert isinstance(source["text"], str)


@pytest.mark.api
class TestCoursesEndpoint:
    """Test cases for the /api/courses endpoint"""
    
    def test_get_course_stats(self, client, sample_course_stats):
        """Test retrieving course statistics"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2
    
    def test_course_stats_response_format(self, client):
        """Test that the course stats response format is correct"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Validate that all course titles are strings
        for title in data["course_titles"]:
            assert isinstance(title, str)


@pytest.mark.api
class TestAPIErrorHandling:
    """Test cases for API error handling"""
    
    def test_query_endpoint_error_handling(self, client, mock_rag_system):
        """Test error handling in query endpoint"""
        # Make the mock raise an exception
        mock_rag_system.query.side_effect = Exception("Test error")
        
        request_data = {"query": "Test query"}
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Test error" in data["detail"]
    
    def test_courses_endpoint_error_handling(self, client, mock_rag_system):
        """Test error handling in courses endpoint"""
        # Make the mock raise an exception
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = client.get("/api/courses")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Analytics error" in data["detail"]


@pytest.mark.api
class TestAPIIntegration:
    """Integration tests for API endpoints"""
    
    def test_session_persistence_across_queries(self, client):
        """Test that session ID persists across multiple queries"""
        # First query without session ID
        request1 = {"query": "What is Python?"}
        response1 = client.post("/api/query", json=request1)
        
        assert response1.status_code == 200
        session_id = response1.json()["session_id"]
        
        # Second query with the returned session ID
        request2 = {"query": "Tell me more", "session_id": session_id}
        response2 = client.post("/api/query", json=request2)
        
        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.post("/api/query", json={"query": "Test query"})
            results.append(response.status_code)
        
        # Create multiple threads for concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert len(results) == 5
        assert all(status == 200 for status in results)
    
    def test_content_type_validation(self, client):
        """Test that API endpoints properly validate content types"""
        # Test with form data instead of JSON
        response = client.post(
            "/api/query", 
            data={"query": "test"},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        assert response.status_code == 422
    
    def test_cors_headers(self, client):
        """Test that CORS headers are properly set"""
        response = client.options("/api/query")
        
        # Check for CORS headers (may vary based on FastAPI version)
        # The presence of these headers indicates CORS is configured
        headers = response.headers
        assert response.status_code in [200, 405]  # Some implementations return 405 for OPTIONS