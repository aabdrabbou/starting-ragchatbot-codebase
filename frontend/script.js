// API base URL - use relative path to work from any host
const API_URL = '/api';

// Global state
let currentSessionId = null;

// DOM elements
let chatMessages, chatInput, sendButton, totalCourses, courseTitles, newChatButton, statusIndicator;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements after page loads
    chatMessages = document.getElementById('chatMessages');
    chatInput = document.getElementById('chatInput');
    sendButton = document.getElementById('sendButton');
    totalCourses = document.getElementById('totalCourses');
    courseTitles = document.getElementById('courseTitles');
    newChatButton = document.getElementById('newChatButton');
    statusIndicator = document.getElementById('statusIndicator');
    
    setupEventListeners();
    createNewSession();
    loadCourseStats();
});

// Event Listeners
function setupEventListeners() {
    // Chat functionality
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    
    
    // New chat button
    if (newChatButton) {
        newChatButton.addEventListener('click', startNewChat);
    }
    
    // Help button
    const helpButton = document.getElementById('helpButton');
    if (helpButton) {
        helpButton.addEventListener('click', showHelpMessage);
    }
    
    // Suggested questions
    document.querySelectorAll('.suggested-item').forEach(button => {
        button.addEventListener('click', (e) => {
            const question = e.target.getAttribute('data-question');
            chatInput.value = question;
            sendMessage();
        });
    });
}


// Chat Functions
async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Disable input
    chatInput.value = '';
    chatInput.disabled = true;
    sendButton.disabled = true;

    // Update status to thinking
    updateStatus('thinking', 'Thinking...');

    // Add user message
    addMessage(query, 'user');

    // Add loading message - create a unique container for it
    const loadingMessage = createLoadingMessage();
    chatMessages.appendChild(loadingMessage);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                session_id: currentSessionId
            })
        });

        if (!response.ok) throw new Error('Query failed');

        const data = await response.json();
        
        // Update session ID if new
        if (!currentSessionId) {
            currentSessionId = data.session_id;
        }

        // Replace loading message with response
        loadingMessage.remove();
        addMessage(data.answer, 'assistant', data.sources);
        updateStatus('online', 'Online');

    } catch (error) {
        // Replace loading message with error
        loadingMessage.remove();
        addMessage(`Error: ${error.message}`, 'assistant');
        updateStatus('offline', 'Error');
        
        // Reset to online after a delay
        setTimeout(() => updateStatus('online', 'Online'), 3000);
    } finally {
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatInput.focus();
    }
}

function createLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant typing-message';
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-text">
                    <span class="typing-avatar">🤖</span>
                    <span class="typing-label">AI is thinking</span>
                </div>
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </div>
    `;
    return messageDiv;
}

function updateStatus(status, text) {
    if (!statusIndicator) return;
    
    statusIndicator.className = `status-indicator ${status}`;
    const statusText = statusIndicator.querySelector('.status-text');
    if (statusText) {
        statusText.textContent = text;
    }
}

function addMessage(content, type, sources = null, isWelcome = false) {
    const messageId = Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}${isWelcome ? ' welcome-message' : ''}`;
    messageDiv.id = `message-${messageId}`;
    
    // Convert markdown to HTML for assistant messages
    const displayContent = type === 'assistant' ? marked.parse(content) : escapeHtml(content);
    
    // Create timestamp
    const timestamp = new Date().toLocaleTimeString('en-US', { 
        hour: 'numeric', 
        minute: '2-digit',
        hour12: true 
    });
    
    let html = `<div class="message-content">${displayContent}</div>
                <div class="message-meta">
                    <span class="message-timestamp">${timestamp}</span>
                    ${type === 'assistant' ? `<button class="copy-button" onclick="copyMessage('${messageId}')" title="Copy message">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                            <path d="m5 15-4-4 4-4"></path>
                        </svg>
                    </button>` : ''}
                </div>`;
    
    if (sources && sources.length > 0) {
        // Convert sources to HTML with clickable links where available
        const sourceElements = sources.map(source => {
            if (source.link) {
                // Create clickable link that opens in new tab with enhanced styling
                return `<div class="source-item clickable">
                    <a href="${escapeHtml(source.link)}" target="_blank" rel="noopener noreferrer">
                        <span class="source-icon">🎥</span>
                        <span class="source-text">${escapeHtml(source.text)}</span>
                        <span class="external-link-icon">↗</span>
                    </a>
                </div>`;
            } else {
                // Plain text for sources without links
                return `<div class="source-item">
                    <span class="source-icon">📄</span>
                    <span class="source-text">${escapeHtml(source.text)}</span>
                </div>`;
            }
        });
        
        html += `
            <details class="sources-collapsible">
                <summary class="sources-header">Sources (${sources.length})</summary>
                <div class="sources-content">${sourceElements.join('')}</div>
            </details>
        `;
    }
    
    messageDiv.innerHTML = html;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageId;
}

// Helper function to escape HTML for user messages
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Copy message functionality
function copyMessage(messageId) {
    const messageDiv = document.getElementById(`message-${messageId}`);
    if (!messageDiv) return;
    
    const contentDiv = messageDiv.querySelector('.message-content');
    if (!contentDiv) return;
    
    // Get text content without HTML tags
    const textContent = contentDiv.innerText || contentDiv.textContent;
    
    navigator.clipboard.writeText(textContent).then(() => {
        // Show feedback
        const copyButton = messageDiv.querySelector('.copy-button');
        if (copyButton) {
            const originalTitle = copyButton.title;
            copyButton.title = 'Copied!';
            copyButton.style.color = 'var(--primary-color)';
            
            setTimeout(() => {
                copyButton.title = originalTitle;
                copyButton.style.color = '';
            }, 2000);
        }
    }).catch(err => {
        console.error('Failed to copy text: ', err);
    });
}

function showHelpMessage() {
    const helpContent = `# 💡 How to Use the Course Materials Assistant

**🎯 Getting Started:**
- Type your questions in the input box below
- Click suggested questions for quick exploration
- Browse available courses in the sidebar

**🔍 What You Can Ask:**
- **Course overviews**: *"What is the MCP course about?"*
- **Specific lessons**: *"What's covered in lesson 3 of [course]?"*
- **Topic searches**: *"Which courses cover RAG?"*
- **Comparisons**: *"Compare the AI courses available"*

**✨ Pro Tips:**
- Click on course titles in the sidebar for quick queries
- Use the copy button to save AI responses
- Sources are provided for all answers - click to explore
- Start a new chat anytime with the "New Chat" button

**🚀 Features:**
- Real-time AI responses with typing indicators
- Source attribution for all information
- Conversation timestamps for reference
- Copy functionality for easy sharing

Happy exploring! 🎉`;

    addMessage(helpContent, 'assistant', null, true);
    chatInput.focus();
}

// Removed removeMessage function - no longer needed since we handle loading differently

async function createNewSession() {
    currentSessionId = null;
    chatMessages.innerHTML = '';
    addMessage(getWelcomeMessage(), 'assistant', null, true);
}

function getWelcomeMessage() {
    return `# 👋 Welcome to Course Materials Assistant!

I'm your AI-powered guide to explore course content intelligently. Here's what I can help you with:

**🎯 What I can do:**
- Find specific information across all course materials
- Explain concepts and provide detailed lesson breakdowns  
- Compare different courses and their content
- Answer questions about instructors and course structure

**💡 Try asking me:**
- *"What courses cover machine learning?"*
- *"Explain the main concepts in lesson 3 of [course name]"*
- *"Who teaches the advanced retrieval course?"*
- *"Compare the MCP course with other AI development courses"*

**🚀 Quick tip:** Use the suggested questions in the sidebar or browse the ${document.getElementById('totalCourses')?.textContent || 'available'} courses to get started!

What would you like to explore first?`;
}

function startNewChat() {
    currentSessionId = null;
    chatMessages.innerHTML = '';
    addMessage(getWelcomeMessage(), 'assistant', null, true);
    chatInput.focus();
}

// Load course statistics
async function loadCourseStats() {
    try {
        console.log('Loading course stats...');
        const response = await fetch(`${API_URL}/courses`);
        if (!response.ok) throw new Error('Failed to load course stats');
        
        const data = await response.json();
        console.log('Course data received:', data);
        
        // Update stats in UI
        if (totalCourses) {
            totalCourses.textContent = data.total_courses;
        }
        
        // Update course titles with click functionality
        if (courseTitles) {
            if (data.course_titles && data.course_titles.length > 0) {
                courseTitles.innerHTML = data.course_titles
                    .map(title => `<div class="course-title-item clickable-course" data-course-title="${escapeHtml(title)}" title="Click to ask about this course">${title}</div>`)
                    .join('');
                
                // Add click listeners to course titles
                courseTitles.querySelectorAll('.clickable-course').forEach(courseItem => {
                    courseItem.addEventListener('click', (e) => {
                        const courseTitle = e.target.getAttribute('data-course-title');
                        const question = `Tell me about the "${courseTitle}" course.`;
                        chatInput.value = question;
                        sendMessage();
                    });
                });
            } else {
                courseTitles.innerHTML = '<span class="no-courses">No courses available</span>';
            }
        }
        
    } catch (error) {
        console.error('Error loading course stats:', error);
        // Set default values on error
        if (totalCourses) {
            totalCourses.textContent = '0';
        }
        if (courseTitles) {
            courseTitles.innerHTML = '<span class="error">Failed to load courses</span>';
        }
    }
}