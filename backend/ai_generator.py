import anthropic
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class RoundState:
    """Tracks conversation state across multiple tool calling rounds"""
    round_number: int = 1
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_execution_count: int = 0
    last_response: Optional[Any] = None
    termination_reason: Optional[str] = None
    
    def add_message(self, message: Dict[str, Any]):
        """Add a message to the conversation"""
        self.messages.append(message)
    
    def should_continue(self, max_rounds: int = 2) -> bool:
        """Check if we should continue with another round"""
        if self.termination_reason:
            return False
        return self.round_number < max_rounds


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search and outline tools for course information.

Tool Usage:
- **Content Search Tool**: Use for questions about specific course content or detailed educational materials
- **Course Outline Tool**: Use for requests about course structure, outlines, or lesson listings
- **Sequential tool usage**: You can make multiple tool calls across up to 2 conversation rounds to gather comprehensive information
- Use tool results to inform follow-up tool calls if needed for complex queries
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Use appropriate tool(s) to gather information, then answer
- **Complex queries**: Use sequential tool calls when you need to:
  - Compare information from different courses or lessons
  - First get an outline/structure, then search specific content
  - Refine search based on initial results
- **Course outline queries**: Use the outline tool and return the course title, course link, and complete list of lessons (number and title for each)
- **Reasoning between tools**: You may briefly reason about tool results to determine if additional searches are needed
- **Direct answers**: Provide clear, comprehensive responses without unnecessary meta-commentary

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
5. **Complete** - Use multiple tool calls if needed to provide thorough answers
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional sequential tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Initialize round state for potential multi-round conversation
        round_state = RoundState()
        round_state.add_message({"role": "user", "content": query})
        
        # Handle up to 2 rounds of tool calling
        max_rounds = 2
        while round_state.should_continue(max_rounds):
            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": round_state.messages.copy(),
                "system": system_content
            }
            
            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}
            
            try:
                # Get response from Claude
                response = self.client.messages.create(**api_params)
                round_state.last_response = response
                
                # Check if Claude wants to use tools
                if response.stop_reason == "tool_use" and tool_manager:
                    # Execute tools and continue to next round
                    round_state = self._execute_round_tools(response, round_state, tool_manager, tools)
                    round_state.round_number += 1
                else:
                    # No tool use - provide final response
                    round_state.termination_reason = "no_tool_use"
                    if response.content and len(response.content) > 0:
                        content_block = response.content[0]
                        if hasattr(content_block, 'text'):
                            return content_block.text
                    return "Unable to extract response text."
                    
            except Exception as e:
                # Handle API errors gracefully
                round_state.termination_reason = f"api_error: {str(e)}"
                if round_state.round_number == 1:
                    # First round failed - return error
                    return f"I encountered an error processing your query: {str(e)}"
                else:
                    # Later round failed - return best available response
                    if hasattr(round_state.last_response, 'content') and round_state.last_response.content:
                        # Find text content in the response
                        for content_block in round_state.last_response.content:
                            if hasattr(content_block, 'text'):
                                return content_block.text
                    return "I encountered an error while gathering additional information, but cannot provide a complete response."
        
        # Max rounds reached - make final call without tools to get text response
        round_state.termination_reason = "max_rounds_reached"
        
        try:
            # Make final API call without tools to force a text response
            final_params = {
                **self.base_params,
                "messages": round_state.messages.copy(),
                "system": system_content
            }
            # Explicitly do NOT include tools to force Claude to provide a text response
            
            final_response = self.client.messages.create(**final_params)
            # Safely extract text from response
            if final_response.content and len(final_response.content) > 0:
                content_block = final_response.content[0]
                if hasattr(content_block, 'text'):
                    return content_block.text
            
            # Fallback if no text content found - synthesize response from tool results
            return "Based on the search results found, I was unable to provide a complete response. Please try rephrasing your question."
            
        except Exception as e:
            # If final call fails, try to extract text from last response
            if round_state.last_response and hasattr(round_state.last_response, 'content'):
                for content_block in round_state.last_response.content:
                    if hasattr(content_block, 'text'):
                        return content_block.text
            
            return f"I encountered an error generating the final response: {str(e)}"
    
    def _execute_round_tools(self, response, round_state: RoundState, tool_manager, tools: Optional[List]) -> RoundState:
        """
        Execute tools for the current round and update conversation state.
        
        Args:
            response: The response containing tool use requests
            round_state: Current conversation state
            tool_manager: Manager to execute tools
            tools: Available tools for potential next round
            
        Returns:
            Updated RoundState with tool execution results
        """
        # Add AI's tool use response to messages
        round_state.add_message({"role": "assistant", "content": response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
                    round_state.tool_execution_count += 1
                    
                except Exception as e:
                    # Handle tool execution errors
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Tool execution failed: {str(e)}"
                    })
                    round_state.termination_reason = f"tool_error: {str(e)}"
        
        # Add tool results as single message
        if tool_results:
            round_state.add_message({"role": "user", "content": tool_results})
        
        return round_state
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Legacy method for backward compatibility with existing single-round tool execution.
        This method is kept for compatibility but the new sequential approach is used in generate_response().
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools (legacy behavior)
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text