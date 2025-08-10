# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Course Materials RAG (Retrieval-Augmented Generation) System that allows users to query course materials and receive AI-powered, context-aware responses. The system uses ChromaDB for vector storage, Anthropic's Claude for AI generation, and provides a web interface for interaction.

## Development Commands

### Running the Application
```bash
# Quick start using the shell script
chmod +x run.sh && ./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Package Management
```bash
# Install dependencies
uv sync

# Add new dependencies (modify pyproject.toml)
uv add package_name
```

### Code Quality Tools
```bash
# Format code automatically
./scripts/format_code.sh

# Run quality checks (formatting and linting)
./scripts/quality_check.sh

# Manual formatting with Black
uv run black backend/ main.py

# Manual linting with Ruff
uv run ruff check backend/ main.py
uv run ruff check --fix backend/ main.py  # Auto-fix issues

# Format with Ruff
uv run ruff format backend/ main.py
```

### Environment Setup
Create a `.env` file in the root directory with:
```
ANTHROPIC_API_KEY=
```

## Architecture

### Core Components

The system follows a modular RAG architecture with these key components:

1. **RAGSystem** (`backend/rag_system.py`) - Main orchestrator that coordinates all components
2. **VectorStore** (`backend/vector_store.py`) - ChromaDB-based vector storage with dual collections:
   - `course_catalog`: Course metadata and titles for semantic search
   - `course_content`: Actual course content chunks
3. **DocumentProcessor** (`backend/document_processor.py`) - Processes course documents into chunks
4. **AIGenerator** (`backend/ai_generator.py`) - Handles Anthropic Claude API interactions with tool support
5. **SessionManager** (`backend/session_manager.py`) - Manages conversation history
6. **SearchTools** (`backend/search_tools.py`) - Tool-based search system for AI agent

### Data Models

- **Course** - Represents a complete course with title, instructor, and lessons
- **Lesson** - Individual lesson within a course with number and title
- **CourseChunk** - Text chunks for vector storage with course/lesson metadata

### Configuration

All settings are centralized in `backend/config.py`:
- Chunk size: 800 characters with 100 character overlap
- Embedding model: `all-MiniLM-L6-v2`
- AI model: `claude-sonnet-4-20250514`
- ChromaDB path: `./chroma_db`

### Tool-Based Search System

The system uses a tool-based approach where the AI can call search functions:
- **CourseSearchTool** - Provides semantic search across course content
- **ToolManager** - Manages tool registration and execution
- Tools return structured results with source attribution

### API Structure

FastAPI application (`backend/app.py`) with endpoints:
- `POST /api/query` - Process user queries
- `GET /api/courses` - Get course statistics
- Static file serving for frontend at `/`

### Document Processing Flow

1. Documents are processed into Course objects with lesson structure
2. Course metadata is added to the catalog collection
3. Content is chunked and stored in the content collection
4. Duplicate courses are detected and skipped based on titles

### Frontend

Simple web interface (`frontend/`) with:
- `index.html` - Main chat interface
- `script.js` - API interaction and UI management
- `style.css` - Styling

## Key Implementation Details

- The system automatically loads documents from the `docs/` folder on startup
- Uses session-based conversation history (max 2 exchanges)
- Implements CORS for cross-origin requests
- ChromaDB collections use sentence transformer embeddings
- Tool-based search allows the AI to dynamically query course content
- Sources are tracked and returned from tool executions
- Always use uv to run the server, don't use pip directly
- make sure you use uv to manage all dependencies
- always use uv to run python files