import os
import sys
import unittest
from unittest.mock import Mock, patch

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_generator import AIGenerator, RoundState
from search_tools import ToolManager


class MockAnthropicResponse:
    """Mock Anthropic API response"""

    def __init__(self, content_text=None, stop_reason="end_turn", tool_calls=None):
        self.stop_reason = stop_reason
        if tool_calls:
            # Tool use response
            self.content = tool_calls
        else:
            # Text response
            mock_content = Mock()
            mock_content.text = content_text or "Test response"
            self.content = [mock_content]


class MockToolUse:
    """Mock tool use content block"""

    def __init__(self, name, input_args, tool_id="tool_123"):
        self.type = "tool_use"
        self.name = name
        self.input = input_args
        self.id = tool_id


class TestAIGenerator(unittest.TestCase):
    """Test suite for AIGenerator tool calling functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test_api_key"
        self.model = "claude-3-sonnet-20240229"
        self.ai_generator = AIGenerator(self.api_key, self.model)

        # Mock tool manager
        self.mock_tool_manager = Mock(spec=ToolManager)

        # Mock tools list
        self.mock_tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for",
                        },
                        "course_name": {
                            "type": "string",
                            "description": "Course filter",
                        },
                        "lesson_number": {
                            "type": "integer",
                            "description": "Lesson filter",
                        },
                    },
                    "required": ["query"],
                },
            }
        ]

    @patch("anthropic.Anthropic")
    def test_generate_response_without_tools(self, mock_anthropic_client):
        """Test basic response generation without tools"""
        # Mock client and response
        mock_client = Mock()
        mock_anthropic_client.return_value = mock_client
        mock_response = MockAnthropicResponse("This is a basic response")
        mock_client.messages.create.return_value = mock_response

        # Create AI generator instance
        ai_gen = AIGenerator(self.api_key, self.model)

        # Generate response
        result = ai_gen.generate_response("What is AI?")

        # Verify response
        self.assertEqual(result, "This is a basic response")

        # Verify API call parameters
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]

        self.assertEqual(call_args["model"], self.model)
        self.assertEqual(call_args["temperature"], 0)
        self.assertEqual(call_args["max_tokens"], 800)
        self.assertEqual(
            call_args["messages"], [{"role": "user", "content": "What is AI?"}]
        )
        self.assertIn("specialized in course materials", call_args["system"])

    @patch("anthropic.Anthropic")
    def test_generate_response_with_conversation_history(self, mock_anthropic_client):
        """Test response generation with conversation history"""
        # Mock client and response
        mock_client = Mock()
        mock_anthropic_client.return_value = mock_client
        mock_response = MockAnthropicResponse("Response with history")
        mock_client.messages.create.return_value = mock_response

        # Create AI generator instance
        ai_gen = AIGenerator(self.api_key, self.model)

        # Generate response with history
        history = "User: Previous question\nAI: Previous answer"
        _result = ai_gen.generate_response(
            "Follow up question", conversation_history=history
        )

        # Verify history is included in system prompt
        call_args = mock_client.messages.create.call_args[1]
        self.assertIn("Previous conversation:", call_args["system"])
        self.assertIn("Previous question", call_args["system"])
        self.assertIn("Previous answer", call_args["system"])

    @patch("anthropic.Anthropic")
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic_client):
        """Test response generation with tools available but no tool use"""
        # Mock client and response
        mock_client = Mock()
        mock_anthropic_client.return_value = mock_client
        mock_response = MockAnthropicResponse("Direct answer without tool use")
        mock_client.messages.create.return_value = mock_response

        # Create AI generator instance
        ai_gen = AIGenerator(self.api_key, self.model)

        # Generate response with tools
        result = ai_gen.generate_response(
            "What is 2+2?", tools=self.mock_tools, tool_manager=self.mock_tool_manager
        )

        # Verify response
        self.assertEqual(result, "Direct answer without tool use")

        # Verify tools were included in API call
        call_args = mock_client.messages.create.call_args[1]
        self.assertEqual(call_args["tools"], self.mock_tools)
        self.assertEqual(call_args["tool_choice"], {"type": "auto"})

        # Verify tool manager was not called
        self.mock_tool_manager.execute_tool.assert_not_called()

    @patch("anthropic.Anthropic")
    def test_generate_response_with_tool_use(self, mock_anthropic_client):
        """Test response generation that uses tools"""
        # Mock client
        mock_client = Mock()
        mock_anthropic_client.return_value = mock_client

        # Mock initial tool use response
        tool_use = MockToolUse(
            name="search_course_content",
            input_args={"query": "machine learning", "course_name": "AI Fundamentals"},
        )
        initial_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_calls=[tool_use]
        )

        # Mock final response after tool execution
        final_response = MockAnthropicResponse(
            "Based on the course content, machine learning is..."
        )

        # Configure mock client to return different responses
        mock_client.messages.create.side_effect = [initial_response, final_response]

        # Mock tool manager execution
        self.mock_tool_manager.execute_tool.return_value = (
            "Machine learning content from course"
        )

        # Create AI generator instance
        ai_gen = AIGenerator(self.api_key, self.model)

        # Generate response
        result = ai_gen.generate_response(
            "What is machine learning?",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager,
        )

        # Verify final response
        self.assertEqual(result, "Based on the course content, machine learning is...")

        # Verify tool was executed
        self.mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="machine learning",
            course_name="AI Fundamentals",
        )

        # Verify at least one API call was made (sequential calling may vary)
        self.assertGreaterEqual(mock_client.messages.create.call_count, 1)

    @patch("anthropic.Anthropic")
    def test_generate_response_multiple_tool_calls(self, mock_anthropic_client):
        """Test response generation with multiple tool calls"""
        # Mock client
        mock_client = Mock()
        mock_anthropic_client.return_value = mock_client

        # Mock multiple tool uses
        tool_use_1 = MockToolUse(
            name="search_course_content",
            input_args={"query": "neural networks"},
            tool_id="tool_1",
        )
        tool_use_2 = MockToolUse(
            name="search_course_content",
            input_args={"query": "deep learning"},
            tool_id="tool_2",
        )

        initial_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_calls=[tool_use_1, tool_use_2]
        )
        final_response = MockAnthropicResponse(
            "Combined response from multiple searches"
        )

        mock_client.messages.create.side_effect = [initial_response, final_response]

        # Mock tool executions
        self.mock_tool_manager.execute_tool.side_effect = [
            "Neural networks content",
            "Deep learning content",
        ]

        # Create AI generator instance
        ai_gen = AIGenerator(self.api_key, self.model)

        # Generate response
        result = ai_gen.generate_response(
            "Explain neural networks and deep learning",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager,
        )

        # Verify result
        self.assertEqual(result, "Combined response from multiple searches")

        # Verify both tools were executed
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 2)

        # Verify the tool calls
        call_args_list = self.mock_tool_manager.execute_tool.call_args_list
        self.assertEqual(call_args_list[0][1], {"query": "neural networks"})
        self.assertEqual(call_args_list[1][1], {"query": "deep learning"})

    @patch("anthropic.Anthropic")
    def test_handle_tool_execution_message_flow(self, mock_anthropic_client):
        """Test correct message flow during tool execution"""
        # Mock client
        mock_client = Mock()
        mock_anthropic_client.return_value = mock_client

        # Mock tool use
        tool_use = MockToolUse(
            name="search_course_content", input_args={"query": "test query"}
        )

        initial_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_calls=[tool_use]
        )
        final_response = MockAnthropicResponse("Final answer")

        mock_client.messages.create.side_effect = [initial_response, final_response]
        self.mock_tool_manager.execute_tool.return_value = "Tool result"

        # Create AI generator instance
        ai_gen = AIGenerator(self.api_key, self.model)

        # Generate response
        result = ai_gen.generate_response(
            "Test query", tools=self.mock_tools, tool_manager=self.mock_tool_manager
        )

        # Verify that tools were executed and result was returned
        self.assertEqual(result, "Final answer")

        # Verify tool was executed
        self.mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="test query"
        )

    @patch("anthropic.Anthropic")
    def test_system_prompt_structure(self, mock_anthropic_client):
        """Test that system prompt contains required instructions"""
        # Mock client and response
        mock_client = Mock()
        mock_anthropic_client.return_value = mock_client
        mock_response = MockAnthropicResponse("Test response")
        mock_client.messages.create.return_value = mock_response

        # Create AI generator instance
        ai_gen = AIGenerator(self.api_key, self.model)

        # Generate response
        ai_gen.generate_response("Test query", tools=self.mock_tools)

        # Check system prompt content
        call_args = mock_client.messages.create.call_args[1]
        system_prompt = call_args["system"]

        # Verify key instructions are present
        self.assertIn("specialized in course materials", system_prompt)
        self.assertIn("Content Search Tool", system_prompt)
        self.assertIn("Course Outline Tool", system_prompt)
        self.assertIn("Sequential tool usage", system_prompt)
        self.assertIn("Brief, Concise and focused", system_prompt)
        self.assertIn("Educational", system_prompt)

    def test_ai_generator_initialization(self):
        """Test AIGenerator initialization and configuration"""
        ai_gen = AIGenerator("test_key", "claude-3-sonnet")

        # Verify configuration
        self.assertEqual(ai_gen.model, "claude-3-sonnet")
        self.assertEqual(ai_gen.base_params["model"], "claude-3-sonnet")
        self.assertEqual(ai_gen.base_params["temperature"], 0)
        self.assertEqual(ai_gen.base_params["max_tokens"], 800)

    def test_round_state_initialization(self):
        """Test RoundState dataclass initialization and methods"""
        state = RoundState()

        # Verify default values
        self.assertEqual(state.round_number, 1)
        self.assertEqual(len(state.messages), 0)
        self.assertEqual(state.tool_execution_count, 0)
        self.assertIsNone(state.last_response)
        self.assertIsNone(state.termination_reason)

        # Test should_continue
        self.assertTrue(state.should_continue(max_rounds=2))

        # Test add_message
        test_message = {"role": "user", "content": "test"}
        state.add_message(test_message)
        self.assertEqual(len(state.messages), 1)
        self.assertEqual(state.messages[0], test_message)

        # Test termination
        state.termination_reason = "test_termination"
        self.assertFalse(state.should_continue(max_rounds=2))

    @patch("anthropic.Anthropic")
    def test_sequential_tool_calling_two_rounds(self, mock_anthropic_client):
        """Test sequential tool calling across two rounds"""
        # Mock client
        mock_client = Mock()
        mock_anthropic_client.return_value = mock_client

        # Round 1: Tool use response
        tool_use_1 = MockToolUse(
            name="get_course_outline",
            input_args={"course_title": "MCP Course"},
            tool_id="tool_1",
        )
        round1_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_calls=[tool_use_1]
        )

        # Round 2: Another tool use response
        tool_use_2 = MockToolUse(
            name="search_course_content",
            input_args={"query": "lesson 3 implementation"},
            tool_id="tool_2",
        )
        round2_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_calls=[tool_use_2]
        )

        # Round 3: Final response (no tools)
        final_response = MockAnthropicResponse(
            stop_reason="end_turn",
            content_text="Based on the course outline and lesson 3 content...",
        )

        # Configure mock client responses
        mock_client.messages.create.side_effect = [
            round1_response,
            round2_response,
            final_response,
        ]

        # Mock tool executions
        self.mock_tool_manager.execute_tool.side_effect = [
            "Course outline with lesson 3: Advanced Implementation",
            "Lesson 3 detailed implementation content",
        ]

        # Create AI generator instance
        ai_gen = AIGenerator(self.api_key, self.model)

        # Execute sequential tool calling
        result = ai_gen.generate_response(
            "Compare lesson 3 of MCP course with implementation details",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager,
        )

        # Verify final response
        self.assertEqual(result, "Based on the course outline and lesson 3 content...")

        # Verify both tools were executed in sequence
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 2)

        # Verify tool execution order and parameters
        call_args_list = self.mock_tool_manager.execute_tool.call_args_list
        self.assertEqual(call_args_list[0][0], "get_course_outline")
        self.assertEqual(call_args_list[0][1], {"course_title": "MCP Course"})
        self.assertEqual(call_args_list[1][0], "search_course_content")
        self.assertEqual(call_args_list[1][1], {"query": "lesson 3 implementation"})

        # Verify multiple API calls were made
        self.assertEqual(mock_client.messages.create.call_count, 3)

    @patch("anthropic.Anthropic")
    def test_sequential_tool_calling_max_rounds_termination(
        self, mock_anthropic_client
    ):
        """Test termination after maximum rounds"""
        # Mock client
        mock_client = Mock()
        mock_anthropic_client.return_value = mock_client

        # Both rounds have tool use (forcing max rounds)
        tool_use = MockToolUse(
            name="search_course_content", input_args={"query": "test"}, tool_id="tool_1"
        )
        tool_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_calls=[tool_use]
        )

        # Configure mock client to always return tool use
        mock_client.messages.create.return_value = tool_response

        # Mock tool execution
        self.mock_tool_manager.execute_tool.return_value = "Tool result"

        # Create AI generator instance
        ai_gen = AIGenerator(self.api_key, self.model)

        # Execute - should terminate after 2 rounds
        result = ai_gen.generate_response(
            "Test query requiring multiple rounds",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager,
        )

        # Should get max rounds termination message
        self.assertIn("unable to generate a complete response", result)

        # Should have made exactly 2 API calls (max rounds)
        self.assertEqual(mock_client.messages.create.call_count, 2)

        # Should have executed tools twice
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 2)

    @patch("anthropic.Anthropic")
    def test_sequential_tool_calling_early_termination(self, mock_anthropic_client):
        """Test early termination when Claude decides to stop using tools"""
        # Mock client
        mock_client = Mock()
        mock_anthropic_client.return_value = mock_client

        # Round 1: Tool use
        tool_use = MockToolUse(
            name="search_course_content",
            input_args={"query": "initial search"},
            tool_id="tool_1",
        )
        round1_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_calls=[tool_use]
        )

        # Round 2: No tool use - direct response
        final_response = MockAnthropicResponse(
            stop_reason="end_turn",
            content_text="I have sufficient information to answer your question.",
        )

        # Configure mock client responses
        mock_client.messages.create.side_effect = [round1_response, final_response]

        # Mock tool execution
        self.mock_tool_manager.execute_tool.return_value = "Search results"

        # Create AI generator instance
        ai_gen = AIGenerator(self.api_key, self.model)

        # Execute
        result = ai_gen.generate_response(
            "Question that Claude can answer after one search",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager,
        )

        # Verify early termination with proper response
        self.assertEqual(
            result, "I have sufficient information to answer your question."
        )

        # Verify only one tool execution
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 1)

        # Verify two API calls (one with tools, one final)
        self.assertEqual(mock_client.messages.create.call_count, 2)

    @patch("anthropic.Anthropic")
    def test_tool_execution_error_handling(self, mock_anthropic_client):
        """Test handling of tool execution errors"""
        # Mock client
        mock_client = Mock()
        mock_anthropic_client.return_value = mock_client

        # Tool use response
        tool_use = MockToolUse(
            name="search_course_content", input_args={"query": "test"}, tool_id="tool_1"
        )
        tool_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_calls=[tool_use]
        )

        mock_client.messages.create.return_value = tool_response

        # Mock tool execution to fail
        self.mock_tool_manager.execute_tool.side_effect = Exception(
            "Tool execution failed"
        )

        # Create AI generator instance
        ai_gen = AIGenerator(self.api_key, self.model)

        # Execute
        result = ai_gen.generate_response(
            "Test query with tool error",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager,
        )

        # Should get error message
        self.assertIn("error processing your query", result)

        # Should have attempted one tool execution
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 1)


if __name__ == "__main__":
    unittest.main()
