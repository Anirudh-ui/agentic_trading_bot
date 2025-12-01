import streamlit as st
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Global Streamlit overrides to match Figma design */
    
    /* Ensure Streamlit elements use the correct theme colors and rounded corners */
    .stButton button[data-testid="baseButton-secondary"] {
        border-color: #93c5fd !important; /* Lighter blue border for secondary button */
    }

    /* Message styling overrides for custom chat bubbles */
    /* The custom CSS below replaces Streamlit's default chat_message styling */
    div[data-testid="stChatMessage"] {
        background-color: transparent !important; /* Resetting Streamlit's default bubble background */
        padding: 0 !important;
        margin: 0 !important;
    }

    .user-message {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 16px 16px 0 16px; /* Custom rounded corners */
        margin: 8px 0;
        max-width: 75%;
        margin-left: auto;
        text-align: left; /* Changed to left for natural text flow */
    }
    
    .bot-message {
        background-color: #f3f4f6;
        color: #111827;
        padding: 12px 16px;
        border-radius: 16px 16px 16px 0; /* Custom rounded corners */
        margin: 8px 0;
        max-width: 75%;
        border: 1px solid #e5e7eb;
    }
    
    .message-time {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-top: 8px;
        display: block;
    }
    
    /* Document card styling (Used for the sidebar buttons) */
    .stButton>button {
        font-size: 0.875rem; /* Smaller font for document details */
        line-height: 1.2;
        padding: 10px 12px;
        text-align: left;
        height: auto;
        white-space: pre-wrap; /* Allows the button text to wrap (for name\nsize) */
    }
    
    .stButton>button[kind="primary"] {
        /* Selected Document */
        background-color: #dbeafe; /* bg-blue-100 */
        color: #1d4ed8; /* text-blue-700 */
        border: 1px solid #93c5fd;
    }
    
    .stButton>button[kind="secondary"] {
        /* Unselected Document */
        background-color: #ffffff; 
        color: #111827;
        border: 1px solid #e5e7eb;
    }
    
    .stButton>button:hover:not([kind="primary"]) {
        background-color: #f3f4f6;
    }

    /* Header styling */
    .chat-header {
        position: sticky;
        top: 0;
        z-index: 10;
        padding: 16px 24px;
        border-bottom: 1px solid #e5e7eb;
        background-color: white;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        color: #9ca3af;
        padding: 48px 24px;
    }
    
    /* General Chat Panel */
    .general-chat-panel {
        background: linear-gradient(135deg, #ecfdf5 0%, #dbeafe 100%);
        padding: 16px;
        border-radius: 12px;
        margin-top: 0; /* Adjusted for better column placement */
        height: 100%;
        border-left: 1px solid #e5e7eb;
    }

    /* Style the General Chat input button */
    div[data-testid="stTextInput"] + div button {
        background-color: #2563eb !important;
        border-color: #2563eb !important;
    }
    
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'selected_doc' not in st.session_state:
    st.session_state.selected_doc = None
if 'doc_messages' not in st.session_state:
    st.session_state.doc_messages = []
if 'general_messages' not in st.session_state:
    st.session_state.general_messages = []
if 'show_general_chat' not in st.session_state:
    st.session_state.show_general_chat = False

# Sidebar - Document Upload and List
with st.sidebar:
    st.title("📄 Documents")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Document",
        type=['pdf', 'doc', 'docx', 'txt'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    # Handle upload and auto-select
    if uploaded_files:
        for file in uploaded_files:
            if not any(doc['name'] == file.name for doc in st.session_state.documents):
                st.session_state.documents.append({
                    'id': len(st.session_state.documents),
                    'name': file.name,
                    'size': f"{file.size / 1024:.1f} KB",
                    'uploaded_at': datetime.now()
                })
                if st.session_state.selected_doc is None:
                    st.session_state.selected_doc = st.session_state.documents[-1]
        st.rerun() # Rerun once after upload processing
            
    st.markdown("---")
    
    # Document list
    if not st.session_state.documents:
        st.markdown("""
        <div class="empty-state" style="padding: 1rem 0;">
            <p>📁</p>
            <p>No documents uploaded</p>
            <p style="font-size: 0.875rem;">Upload documents to get started</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for doc in st.session_state.documents:
            is_selected = (st.session_state.selected_doc and 
                          st.session_state.selected_doc['id'] == doc['id'])
            
            button_text = f"📄 {doc['name']}\n{doc['size']}"

            if st.button(
                button_text,
                key=f"doc_{doc['id']}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.selected_doc = doc
                st.rerun()

# --- MAIN LAYOUT ---

if st.session_state.show_general_chat:
    # Two columns: 7 parts for main chat, 3 parts for general chat (Sum = 10)
    cols = st.columns([7, 3])
    col1 = cols[0]
    col2 = cols[1]
else:
    # One column: 10 parts for main chat (Sum = 10)
    col1 = st.columns([10])[0]
    col2 = None

# --- Document Chat Area (col1) ---
with col1:
    # Header
    if st.session_state.selected_doc:
        st.markdown(f"""
        <div class="chat-header">
            <h2>💬 Chat with Document</h2>
            <p style="color: #6b7280; margin-top: 4px;">{st.session_state.selected_doc['name']}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="chat-header">
            <h2>📝 Document Q&A</h2>
            <p style="color: #6b7280; margin-top: 4px;">Select a document to start chatting</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Messages area
    message_container = st.container(height=500)
    
    if not st.session_state.selected_doc:
        with message_container:
            st.markdown("""
            <div class="empty-state">
                <p style="font-size: 3rem;">📄</p>
                <p>Select a document from the sidebar</p>
                <p style="font-size: 0.875rem; margin-top: 8px;">to start asking questions</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        with message_container:
            if not st.session_state.doc_messages:
                 st.markdown("""
                 <div class="empty-state">
                     <p>Ask questions about the document</p>
                     <p style="font-size: 0.875rem; margin-top: 8px;">The AI will help you find answers</p>
                 </div>
                 """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.doc_messages:
                    if msg['sender'] == 'user':
                        st.markdown(f"""
                        <div class="user-message">
                            <p>{msg['text']}</p>
                            <div class="message-time">{msg['timestamp'].strftime('%I:%M %p')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="bot-message">
                            <p>{msg['text']}</p>
                            <div class="message-time">{msg['timestamp'].strftime('%I:%M %p')}</div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Input area - Split into form (for text/send) and button (for general chat)

    # 1. Document Chat Form (Text Input + Send Button)
    send_clicked = False
    with st.form(key="doc_chat_form", clear_on_submit=True):
        # We only need two columns here: one for text input, one for the form_submit_button
        input_col1, input_col2 = st.columns([6, 1]) 
        
        with input_col1:
            user_input = st.text_input(
                "Message",
                placeholder="Ask a question about the document..." if st.session_state.selected_doc else "Select a document first...",
                disabled=not st.session_state.selected_doc,
                label_visibility="collapsed",
                key="doc_input"
            )
        
        with input_col2:
            send_clicked = st.form_submit_button(
                "📤",
                disabled=not st.session_state.selected_doc or not user_input,
                type="primary",
                use_container_width=True
            )

    # 2. General Chat Button (Outside the form)
    # Use st.columns again to place this button immediately next to the form area
    _, chat_btn_col = st.columns([7, 1])
    
    with chat_btn_col:
        # st.button can now be used because it's outside the form.
        general_chat_btn = st.button(
            "💬",
            key="open_general_chat_btn",
            disabled=st.session_state.show_general_chat,
            help="Open General Chat",
            use_container_width=True
        )

    if general_chat_btn:
        st.session_state.show_general_chat = True
        st.rerun()

    # Handle message sending
    if send_clicked and user_input and st.session_state.selected_doc:
        st.session_state.doc_messages.append({
            'text': user_input,
            'sender': 'user',
            'timestamp': datetime.now()
        })
        
        time.sleep(1)
        bot_response = f"Based on the document \"{st.session_state.selected_doc['name']}\", here's the answer to your question. This is a simulated response."
        
        st.session_state.doc_messages.append({
            'text': bot_response,
            'sender': 'bot',
            'timestamp': datetime.now()
        })
        
        st.rerun()

# --- General Chat Panel (col2) ---
if st.session_state.show_general_chat and col2 is not None:
    with col2:
        # Header and Close Button
        st.markdown("""
        <div class="general-chat-panel" style="padding-top: 0; padding-bottom: 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; padding: 1rem 0 0.5rem 0;">
                <h3 style="margin: 0;">💬 General Chat</h3>
            </div>
            <p style="color: #6b7280; font-size: 0.875rem; margin-bottom: 1rem;">Ask me anything</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("❌ Close", key="close_general", use_container_width=True):
            st.session_state.show_general_chat = False
            st.session_state.general_messages = []
            st.rerun()
        
        # General chat messages
        gen_msg_container = st.container(height=350)
        
        if not st.session_state.general_messages:
            with gen_msg_container:
                st.markdown("""
                <div class="empty-state">
                    <p style="font-size: 2rem;">💬</p>
                    <h4>Welcome to General Chat</h4>
                    <p>Start a conversation</p>
                    <p style="font-size: 0.875rem;">I can help with various topics!</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            with gen_msg_container:
                for msg in st.session_state.general_messages:
                    if msg['sender'] == 'user':
                        st.markdown(f"""
                        <div class="user-message">
                            <p>{msg['text']}</p>
                            <div class="message-time">{msg['timestamp'].strftime('%I:%M %p')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="bot-message">
                            <p>{msg['text']}</p>
                            <div class="message-time">{msg['timestamp'].strftime('%I:%M %p')}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # General chat input
        with st.form(key="general_chat_form", clear_on_submit=True):
            gen_input = st.text_input(
                "General message",
                placeholder="Type your message...",
                label_visibility="collapsed",
                key="gen_input_field"
            )
            
            gen_send_clicked = st.form_submit_button("Send", disabled=not gen_input, use_container_width=True, type="primary")
            
            if gen_send_clicked and gen_input:
                st.session_state.general_messages.append({
                    'text': gen_input,
                    'sender': 'user',
                    'timestamp': datetime.now()
                })
                
                time.sleep(1)
                st.session_state.general_messages.append({
                    'text': "This is a general response to your query. I can help with various topics!",
                    'sender': 'bot',
                    'timestamp': datetime.now()
                })
                
                st.rerun()