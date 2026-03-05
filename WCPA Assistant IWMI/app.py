from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import os
import json
import logging
from datetime import datetime
import sys
import traceback
from typing import Dict, List, Any
import secrets
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import the RAG pipeline
try:
    from rag_pipeline2 import RAGPipeline2 as RAGPipeline
    RAG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"RAG pipeline import failed: {e}. Running in limited mode.")
    RAG_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='dist', static_url_path='')
CORS(app)

# Configuration
INDEX_FILE = "pdf_index_enhanced1.pkl"

# OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:5173/auth/callback")

# Model configuration - Load from environment or use defaults
LLM_TYPE = os.getenv("LLM_TYPE", "deepseek").lower()

# Get Hugging Face tokens and Google API key from environment
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_BACKUP_TOKEN_1 = os.getenv("HF_BACKUP_TOKEN_1", "")
HF_BACKUP_TOKEN_2 = os.getenv("HF_BACKUP_TOKEN_2", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Azure configuration (if using)
AZURE_KEY = os.getenv("AZURE_KEY", "")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-4o-mini")

# Session management
sessions: Dict[str, Dict] = {}

# Initialize RAG pipeline
rag_pipeline = None

def initialize_rag_pipeline():
    """Initialize the RAG pipeline with configured model parameters"""
    global rag_pipeline
    
    model_params = {
        "llm_type": LLM_TYPE,
        "hf_token": HF_TOKEN,
        "hf_backup_token_1": HF_BACKUP_TOKEN_1,
        "hf_backup_token_2": HF_BACKUP_TOKEN_2,
        "google_api_key": GOOGLE_API_KEY,
    }
    
    if LLM_TYPE == "azure":
        model_params.update({
            "azure_key": AZURE_KEY,
            "azure_endpoint": AZURE_ENDPOINT,
            "azure_deployment": AZURE_DEPLOYMENT,
        })
    else:  # deepseek
        # Use the correct router endpoint
        model_params.update({
            "deepseek_url": "https://router.huggingface.co/v1/chat/completions",
            "deepseek_model": "deepseek-ai/DeepSeek-V3.1",
        })
    
    try:
        # Create a dummy PDF folder path (won't be used since we're loading from pickle)
        # The RAGPipeline class expects a pdf_folder parameter, but it won't be used
        # if load_index() is successful
        dummy_pdf_folder = "./dummy_pdfs"
        
        rag_pipeline = RAGPipeline(
            pdf_folder=dummy_pdf_folder,  # This won't be used since we load from pickle
            index_file=INDEX_FILE,
            model_params=model_params,
            reserve_tokens=8000,
        )
        
        # Load existing index
        if rag_pipeline.load_index():
            logger.info(f"✅ Loaded RAG index with {len(rag_pipeline.documents)} chunks")
        else:
            logger.info("❌ No existing index found. You need to build an index first.")
            
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        logger.error(traceback.format_exc())
        return False

# Initialize on startup
initialize_rag_pipeline()

def get_session(session_id: str) -> Dict:
    """Get or create a session"""
    if session_id not in sessions:
        sessions[session_id] = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],
            "title": f"Wetland Session {len(sessions) + 1}"
        }
    return sessions[session_id]

def update_session_title(session_id: str, first_message: str):
    """Update session title based on first message"""
    if session_id in sessions:
        # Create a short title from first message
        words = first_message.split()[:5]
        title = " ".join(words) + ("..." if len(first_message.split()) > 5 else "")
        sessions[session_id]["title"] = f"🌿 {title}"
        sessions[session_id]["updated_at"] = datetime.now().isoformat()

@app.route('/')
def serve_frontend():
    """Serve the React frontend"""
    try:
        return send_from_directory(app.static_folder, 'index.html')
    except:
        return jsonify({"status": "Frontend not built. Run 'npm run build' first"})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        if rag_pipeline is None:
            return jsonify({
                "status": "initializing",
                "llm_type": LLM_TYPE,
                "index_loaded": False
            })
        
        # Check if index is loaded
        index_loaded = hasattr(rag_pipeline, 'documents') and len(rag_pipeline.documents) > 0
        
        return jsonify({
            "status": "healthy" if index_loaded else "no_index",
            "llm_type": LLM_TYPE,
            "index_loaded": index_loaded,
            "chunks_available": len(rag_pipeline.documents) if index_loaded else 0
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/auth/google/login', methods=['GET'])
def google_login():
    """Get Google OAuth URL"""
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile",
        "access_type": "offline",
        "prompt": "consent"
    }
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{requests.compat.urlencode(params)}"
    return jsonify({"auth_url": auth_url})

@app.route('/api/auth/exchange-token', methods=['POST'])
def exchange_token():
    """Exchange code for tokens"""
    try:
        data = request.get_json()
        code = data.get('code')
        if not code:
            return jsonify({"error": "Code is required"}), 400
        
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code"
        }
        
        response = requests.post(token_url, data=data)
        if not response.ok:
            return jsonify({"error": "Failed to exchange token", "details": response.json()}), 400
            
        return jsonify(response.json())
    except Exception as e:
        logger.error(f"Token exchange error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/auth/userinfo', methods=['GET'])
def get_user_info():
    """Get user info from Google"""
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authorization header missing"}), 401
            
        access_token = auth_header.split(' ')[1]
        userinfo_url = "https://www.googleapis.com/oauth2/v3/userinfo"
        response = requests.get(userinfo_url, headers={"Authorization": f"Bearer {access_token}"})
        
        if not response.ok:
            return jsonify({"error": "Failed to get user info"}), 400
            
        return jsonify(response.json())
    except Exception as e:
        logger.error(f"User info error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        message = data.get('message', '').strip()
        session_id = data.get('session_id', f'session-{secrets.token_hex(8)}')
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        if rag_pipeline is None:
            return jsonify({"error": "RAG pipeline not initialized"}), 500
        
        # Check if index is loaded
        if not hasattr(rag_pipeline, 'documents') or len(rag_pipeline.documents) == 0:
            return jsonify({
                "error": "No index loaded. Please rebuild the index first.",
                "response": "I don't have any documents loaded yet. Please rebuild the index from the admin panel."
            }), 400
        
        # Get or create session
        session = get_session(session_id)
        
        # Update session title if first message
        if len(session['messages']) == 0:
            update_session_title(session_id, message)
        
        # Add user message to session
        user_msg = {
            "id": len(session['messages']),
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        session['messages'].append(user_msg)
        
        logger.info(f"Processing query: {message[:100]}...")
        
        # Query RAG pipeline
        answer = rag_pipeline.query(message, top_k=8)
        docs = getattr(rag_pipeline, "last_retrieved_docs", [])
        
        # Format sources
        sources = []
        for i, doc in enumerate(docs, 1):
            sources.append({
                "rank": i,
                "filename": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 'N/A'),
                "type": doc.metadata.get('type', 'text'),
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "full_content": doc.page_content,
                "metadata": doc.metadata
            })
        
        # Add assistant response to session
        assistant_msg = {
            "id": len(session['messages']),
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }
        session['messages'].append(assistant_msg)
        
        # Update session timestamp
        session['updated_at'] = datetime.now().isoformat()
        
        # Get conversation stats
        stats = rag_pipeline.get_conversation_stats()
        
        return jsonify({
            "response": answer,
            "sources": sources,
            "session_id": session_id,
            "retrieval_stats": {
                "retrieved_chunks": len(docs),
                "total_chunks_available": len(rag_pipeline.documents),
                "conversation_history_tokens": stats.get('history_tokens', 0)
            }
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "response": "Sorry, I encountered an error processing your request."
        }), 500

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Handle chat messages with streaming response"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        message = data.get('message', '').strip()
        session_id = data.get('session_id', f'session-{secrets.token_hex(8)}')
        mode = data.get('mode', 'thinking') # Default to thinking
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        if rag_pipeline is None:
            return jsonify({"error": "RAG pipeline not initialized"}), 500
        
        # Get or create session
        session = get_session(session_id)
        
        # Update session title if first message
        if len(session['messages']) == 0:
            update_session_title(session_id, message)
        
        # Add user message to session
        user_msg = {
            "id": len(session['messages']),
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        session['messages'].append(user_msg)
        
        logger.info(f"Processing query (STREAM) [Mode: {mode}]: {message[:100]}...")

        def generate():
            try:
                yield f"data: {json.dumps({'type': 'session_id', 'content': session_id})}\n\n"
                
                full_answer_text = ""
                for chunk in rag_pipeline.query_stream(message, top_k=8, mode=mode):
                    yield f"data: {chunk}\n\n"
                    try:
                        chunk_data = json.loads(chunk)
                        if chunk_data.get('type') == 'answer':
                            full_answer_text += chunk_data.get('content', '')
                    except:
                        pass
                
                # After stream is done, send sources
                docs = getattr(rag_pipeline, "last_retrieved_docs", [])
                sources = []
                for i, doc in enumerate(docs, 1):
                    sources.append({
                        "rank": i,
                        "filename": doc.metadata.get('source', 'Unknown'),
                        "page": doc.metadata.get('page', 'N/A'),
                        "type": doc.metadata.get('type', 'text'),
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "full_content": doc.page_content,
                        "metadata": doc.metadata
                    })
                
                yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"
                
                # Add assistant response to session
                assistant_msg = {
                    "id": len(session['messages']),
                    "role": "assistant",
                    "content": full_answer_text,
                    "sources": sources,
                    "timestamp": datetime.now().isoformat()
                }
                session['messages'].append(assistant_msg)
                session['updated_at'] = datetime.now().isoformat()
                
                yield "data: [DONE]\n\n"
                logger.info(f"Stream completed for session {session_id}")
            except Exception as e:
                logger.error(f"Error in stream generator: {e}")
                logger.error(traceback.format_exc())
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
                yield "data: [DONE]\n\n"

        headers = {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
        return Response(stream_with_context(generate()), headers=headers)
        
    except Exception as e:
        logger.error(f"Chat stream error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear conversation history for a session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', '')
        
        if not session_id:
            return jsonify({"error": "Session ID is required"}), 400
        
        if rag_pipeline:
            rag_pipeline.clear_conversation()
        
        # Reset session
        if session_id in sessions:
            # Keep only the session metadata, clear messages
            sessions[session_id]['messages'] = []
            sessions[session_id]['updated_at'] = datetime.now().isoformat()
        
        return jsonify({
            "status": "cleared",
            "session_id": session_id
        })
        
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    try:
        # For now, return configured model
        models = [
            {
                "key": "wetland-ai",
                "name": "Wetland Conservation AI",
                "type": "wetland",
                "is_current": True
            },
            {
                "key": "aquatic-research",
                "name": "Aquatic Research Model",
                "type": "research",
                "is_current": False
            }
        ]
        
        return jsonify({
            "available_models": models,
            "current_model": "wetland-ai",
            "llm_backend": LLM_TYPE
        })
        
    except Exception as e:
        logger.error(f"Get models error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/switch-model', methods=['POST'])
def switch_model():
    """Switch between models (placeholder for future implementation)"""
    try:
        data = request.get_json()
        model_key = data.get('model', '')
        
        if not model_key:
            return jsonify({"error": "Model key is required"}), 400
        
        # For now, just log the request
        logger.info(f"Model switch requested to: {model_key}")
        
        return jsonify({
            "status": "switched",
            "model": model_key,
            "model_info": {
                "name": "Wetland Conservation AI",
                "type": "wetland"
            }
        })
        
    except Exception as e:
        logger.error(f"Switch model error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        if rag_pipeline is None:
            return jsonify({
                "status": "initializing",
                "index": {"total_chunks": 0},
                "sessions": {"total": len(sessions)}
            })
        
        # Get RAG pipeline stats
        pipeline_stats = rag_pipeline.get_stats() if rag_pipeline else {}
        
        # Get conversation stats
        conv_stats = rag_pipeline.get_conversation_stats() if rag_pipeline else {}
        
        # Check if index is loaded
        index_loaded = hasattr(rag_pipeline, 'documents') and len(rag_pipeline.documents) > 0
        
        return jsonify({
            "status": "healthy" if index_loaded else "no_index",
            "index": {
                "total_chunks": pipeline_stats.get('total_chunks', 0),
                "content_types": pipeline_stats.get('content_types', {}),
                "loaded": index_loaded
            },
            "conversation": conv_stats,
            "sessions": {
                "total": len(sessions),
                "active": len([s for s in sessions.values() if s['messages']])
            },
            "llm_backend": LLM_TYPE
        })
        
    except Exception as e:
        logger.error(f"Get stats error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sources', methods=['GET'])
def get_sources():
    """Get list of uploaded documents/sources"""
    try:
        # Get sources from RAG pipeline if available
        rag_sources = []
        if rag_pipeline and hasattr(rag_pipeline, 'documents'):
            # Get unique sources
            sources_set = set()
            for doc in rag_pipeline.documents:
                source = doc.metadata.get('source', 'Unknown')
                if source:
                    sources_set.add(source)
            rag_sources = list(sources_set)
        
        return jsonify({
            "sources": rag_sources,
            "total_sources": len(rag_sources)
        })
        
    except Exception as e:
        logger.error(f"Get sources error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get all chat sessions"""
    try:
        session_list = []
        for session_id, session_data in sessions.items():
            session_list.append({
                "session_id": session_id,
                "title": session_data.get('title', 'Untitled Session'),
                "created_at": session_data.get('created_at'),
                "updated_at": session_data.get('updated_at'),
                "message_count": len(session_data.get('messages', [])),
                "preview": session_data.get('messages', [])[0]['content'][:100] + "..." 
                           if session_data.get('messages') else ""
            })
        
        # Sort by updated_at (most recent first)
        session_list.sort(key=lambda x: x['updated_at'], reverse=True)
        
        return jsonify({
            "sessions": session_list,
            "total": len(session_list)
        })
        
    except Exception as e:
        logger.error(f"Get sessions error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session_data(session_id):
    """Get specific session data"""
    try:
        session = get_session(session_id)
        
        return jsonify({
            "session_id": session_id,
            "title": session.get('title', 'Untitled Session'),
            "created_at": session.get('created_at'),
            "updated_at": session.get('updated_at'),
            "messages": session.get('messages', []),
            "message_count": len(session.get('messages', []))
        })
        
    except Exception as e:
        logger.error(f"Get session error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a session"""
    try:
        if session_id in sessions:
            del sessions[session_id]
            
        return jsonify({
            "status": "deleted",
            "session_id": session_id
        })
        
    except Exception as e:
        logger.error(f"Delete session error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/rebuild-index', methods=['POST'])
def rebuild_index():
    """Rebuild the RAG index (admin endpoint) - DISABLED since no PDF folder"""
    try:
        return jsonify({
            "status": "error",
            "message": "Rebuild index is disabled. The index is loaded from pickle file only."
        }), 400
        
    except Exception as e:
        logger.error(f"Rebuild index error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug/chunks', methods=['GET'])
def debug_chunks():
    """Debug endpoint to view chunks"""
    try:
        if rag_pipeline is None:
            return jsonify({"error": "RAG pipeline not initialized"}), 500
        
        source = request.args.get('source', '')
        limit = int(request.args.get('limit', 10))
        
        chunks = []
        for doc in rag_pipeline.documents[:limit]:
            if source and doc.metadata.get('source') != source:
                continue
            
            chunks.append({
                "metadata": doc.metadata,
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "content_length": len(doc.page_content)
            })
        
        return jsonify({
            "chunks": chunks,
            "total_chunks": len(rag_pipeline.documents),
            "showing": len(chunks)
        })
        
    except Exception as e:
        logger.error(f"Debug chunks error: {e}")
        return jsonify({"error": str(e)}), 500

# Serve React static files
@app.route('/<path:path>')
def serve_static(path):
    try:
        return send_from_directory(app.static_folder, path)
    except:
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    # Start Flask server
    port = int(os.getenv("PORT", 5000))
    
    # Check if index file exists
    if not os.path.exists(INDEX_FILE):
        logger.warning(f"Index file {INDEX_FILE} not found. The RAG pipeline will be in no_index state.")
    
    app.run(host='0.0.0.0', port=port, debug=True)