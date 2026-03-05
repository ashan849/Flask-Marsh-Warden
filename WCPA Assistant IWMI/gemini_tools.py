"""
Enhanced Agentic Tools for Gemini API Function Calling
Defines tool schemas and execution functions for the RAG chatbot
"""

import logging
from typing import Dict, List, Any, Optional
import json
import re

logger = logging.getLogger(__name__)

# Tool schemas for Gemini API function calling
TOOL_SCHEMAS = [
    {
        "name": "retrieve_documents",
        "description": (
            "Retrieve relevant context chunks from the wetland conservation knowledge base. "
            "Use this for general thematic search or when you need broad information across all documents. "
            "Best for 'what', 'how', and 'why' questions. "
            "Use top_k=12 or higher for complex, multi-part questions."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "A specific search query. Phrasing it as a question or a set of keywords works best."
                },
                "top_k": {
                    "type": "INTEGER",
                    "description": "Number of top documents to retrieve (default: 12, max: 20). Use higher values for complex questions."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_specific_document",
        "description": (
            "Search within a single specified document. Use this when you have identified a relevant document "
            "from a general search or when the user mentions a specific report (e.g., 'National Wetland Policy'). "
            "Always prefer this over retrieve_documents when you know which document to target."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "document_name": {
                    "type": "STRING",
                    "description": "The exact name of the document as shown in the document list."
                },
                "query": {
                    "type": "STRING",
                    "description": "Specific query to run within this document."
                },
                "top_k": {
                    "type": "INTEGER",
                    "description": "Number of chunks to retrieve (default: 8, max: 15)."
                }
            },
            "required": ["document_name", "query"]
        }
    },
    {
        "name": "get_document_list",
        "description": (
            "Retrieve the names and metadata of all available documents in the knowledge base. "
            "ALWAYS call this first when you are unsure which document contains the answer, "
            "or to get exact document names for search_specific_document."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "verify_answer",
        "description": (
            "CRITICAL: Verify whether a specific factual statement is supported by the knowledge base. "
            "Use this BEFORE writing your Final Answer to double-check key facts, numbers, "
            "penalties, dates, or policy provisions you have extracted. "
            "Returns a verdict: 'supported', 'partial', or 'contradicted', plus the supporting evidence."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "statement": {
                    "type": "STRING",
                    "description": "The specific factual claim to verify (e.g., 'The fine for unauthorized construction is Rs. 500,000')."
                },
                "source_hint": {
                    "type": "STRING",
                    "description": "Optional: name of the document you believe contains the evidence."
                }
            },
            "required": ["statement"]
        }
    },
    {
        "name": "plan_answer_structure",
        "description": (
            "MANDATORY: Propose a custom markdown structure for the final answer. "
            "Use this AFTER gathering enough evidence but BEFORE writing the Final Answer. "
            "You MUST use this to organize complex information into logical sections (headings, tables, bullets). "
            "Use tables for any data involving 3+ comparisons or numerical lists."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "question": {
                    "type": "STRING",
                    "description": "The original user question."
                },
                "key_findings": {
                    "type": "STRING",
                    "description": "A brief summary of the key facts/evidence gathered so far."
                },
                "proposed_sections": {
                    "type": "ARRAY",
                    "items": { "type": "STRING" },
                    "description": "List of markdown heading names and section types (e.g., '## Legal Analysis', 'Table of Penalties')."
                },
                "identified_intent": {
                    "type": "STRING",
                    "description": "The identified user intent: 'Direct' (concise/factual) or 'Informative' (detailed/research)."
                },
                "target_word_count": {
                    "type": "STRING",
                    "description": "The target word count range for the final answer (e.g. '50-150' or '400-500')."
                }
            },
            "required": ["question", "key_findings", "proposed_sections", "identified_intent", "target_word_count"]
        }
    }
]


class ToolExecutor:
    """Executes tool calls for the agentic RAG system"""

    def __init__(self, rag_pipeline):
        """
        Initialize tool executor with RAG pipeline

        Args:
            rag_pipeline: RAGPipeline instance for document retrieval
        """
        self.rag_pipeline = rag_pipeline
        logger.info("ToolExecutor initialized with RAG pipeline")

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call and return results

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool

        Returns:
            Dict with tool execution results
        """
        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

        try:
            if tool_name == "retrieve_documents":
                return self._retrieve_documents(tool_args)
            elif tool_name == "search_specific_document":
                return self._search_specific_document(tool_args)
            elif tool_name == "get_document_list":
                return self._get_document_list(tool_args)
            elif tool_name == "verify_answer":
                return self._verify_answer(tool_args)
            elif tool_name == "plan_answer_structure":
                return self._plan_answer_structure(tool_args)
            else:
                return {
                    "error": f"Unknown tool: {tool_name}",
                    "success": False
                }
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            return {
                "error": str(e),
                "success": False
            }

    def _retrieve_documents(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute retrieve_documents tool"""
        # Ensure top_k is an integer — raised default from 8 → 12
        try:
            top_k = int(args.get("top_k", 12))
        except (ValueError, TypeError):
            top_k = 12

        top_k = min(top_k, 20)  # hard ceiling

        # Ensure query is a string
        query = str(args.get("query", args.get("question", "")))

        if not query:
            return {"error": "Query is required", "success": False}

        # ── Step 1: Retrieval ────────────────────────────────────────────────
        if hasattr(self.rag_pipeline, '_expand_and_retrieve'):
            retrieved_docs = self.rag_pipeline._expand_and_retrieve(query, top_k=top_k)
        elif hasattr(self.rag_pipeline, 'hybrid_retriever'):
            retrieved_docs = self.rag_pipeline.hybrid_retriever.invoke(query)
        else:
            return {"error": "RAG pipeline not initialized", "success": False}

        # ── Step 2: Processing ───────────────────────────────────────────────
        if not retrieved_docs:
            return {
                "success": True,
                "message": "No relevant documents found",
                "documents": [],
                "count": 0
            }

        # Apply relevance filtering
        top_docs = retrieved_docs[:top_k * 2] if len(retrieved_docs) > top_k else retrieved_docs
        filtered = self.rag_pipeline.relevance_checker.filter_documents(query, top_docs)
        
        # Take the top_k requested
        filtered_docs_with_scores = filtered[:top_k]
        filtered_docs = [d for d, s in filtered_docs_with_scores]
        scores = [s for d, s in filtered_docs_with_scores]

        # Format results with confidence hints
        results = []
        for doc, score in zip(filtered_docs, scores):
            confidence_level = "HIGH" if score >= 0.80 else ("MEDIUM" if score >= 0.65 else "LOW")
            results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "?"),
                "type": doc.metadata.get("type", "text"),
                "relevance": confidence_level,
                "score": round(float(score), 4)
            })

        return {
            "success": True,
            "message": f"Retrieved {len(results)} relevant documents",
            "documents": results,
            "count": len(results)
        }

    def _search_specific_document(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search_specific_document tool"""
        document_name = str(args.get("document_name", ""))
        query = str(args.get("query", ""))

        # Ensure top_k is an integer — raised default from 5 → 8
        try:
            top_k = int(args.get("top_k", 8))
        except (ValueError, TypeError):
            top_k = 8

        top_k = min(top_k, 15)  # hard ceiling

        if not document_name or not query:
            return {
                "error": "Both document_name and query are required",
                "success": False
            }

        # ── Step 1: Broad Retrieval ──────────────────────────────────────────
        if hasattr(self.rag_pipeline, '_expand_and_retrieve'):
            # Fetch more (30) to increase chances of finding chunks from the specific doc
            retrieved_docs = self.rag_pipeline._expand_and_retrieve(query, top_k=30)
        elif hasattr(self.rag_pipeline, 'hybrid_retriever'):
            retrieved_docs = self.rag_pipeline.hybrid_retriever.invoke(query)
        else:
            return {"error": "RAG pipeline not initialized", "success": False}

        # ── Step 2: Document Filtering ──────────────────────────────────────
        if not retrieved_docs:
            return {
                "success": True,
                "message": f"No content found in knowledge base for query. Check document names with get_document_list.",
                "documents": [],
                "count": 0,
                "searched_document": document_name
            }

        # Filter by specific document (fuzzy match on source name)
        doc_specific = [
            doc for doc in retrieved_docs
            if document_name.lower() in str(doc.metadata.get("source", "")).lower()
        ]

        if not doc_specific:
            # Try a broader keyword match (first word of document_name)
            first_word = document_name.split()[0].lower() if document_name.split() else ""
            if first_word:
                doc_specific = [
                    doc for doc in retrieved_docs
                    if first_word in str(doc.metadata.get("source", "")).lower()
                ]

        if not doc_specific:
            return {
                "success": True,
                "message": f"Wait! '{document_name}' was not found in the initial retrieval results. Try general retrieve_documents first.",
                "documents": [],
                "count": 0,
                "searched_document": document_name
            }

        # ── Step 3: Relevance Filtering & Formatting ─────────────────────────
        top_docs = doc_specific[:top_k * 2] if len(doc_specific) > top_k else doc_specific
        filtered = self.rag_pipeline.relevance_checker.filter_documents(query, top_docs)
        filtered_docs_with_scores = filtered[:top_k]
        filtered_docs = [d for d, s in filtered_docs_with_scores]
        scores = [s for d, s in filtered_docs_with_scores]

        # Format results
        results = []
        for doc, score in zip(filtered_docs, scores):
            confidence_level = "HIGH" if score >= 0.80 else ("MEDIUM" if score >= 0.65 else "LOW")
            results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "?"),
                "type": doc.metadata.get("type", "text"),
                "relevance": confidence_level,
                "score": round(float(score), 4)
            })

        return {
            "success": True,
            "message": f"Retrieved {len(results)} chunks from '{document_name}'",
            "documents": results,
            "count": len(results),
            "searched_document": document_name
        }

    def _verify_answer(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute verify_answer tool.
        Re-retrieves evidence for a specific factual statement and returns a verdict.
        """
        statement = str(args.get("statement", ""))
        source_hint = str(args.get("source_hint", ""))

        if not statement:
            return {"error": "statement is required", "success": False}

        if not hasattr(self.rag_pipeline, 'hybrid_retriever'):
            return {"error": "RAG pipeline not initialized", "success": False}

        # Use statement as the target query
        query = statement

        # Use multi-query expansion for verification re-retrieval
        if hasattr(self.rag_pipeline, '_expand_and_retrieve'):
            retrieved_docs = self.rag_pipeline._expand_and_retrieve(query, top_k=20)
        elif hasattr(self.rag_pipeline, 'hybrid_retriever'):
            retrieved_docs = self.rag_pipeline.hybrid_retriever.invoke(query)

        # Filter by source hint if provided
        if source_hint:
            target_docs = [
                doc for doc in retrieved_docs
                if source_hint.lower() in doc.metadata.get("source", "").lower()
            ]
            if not target_docs:
                target_docs = retrieved_docs  # fall back to all
        else:
            target_docs = retrieved_docs

        if not target_docs:
            return {
                "success": True,
                "verdict": "unverifiable",
                "confidence": 0,
                "message": "No relevant evidence found in knowledge base to verify this statement.",
                "evidence": []
            }

        # Re-rank and take top 6 chunks for verification
        top_docs = target_docs[:12]
        filtered = self.rag_pipeline.relevance_checker.filter_documents(query, top_docs)
        verified_docs = [d for d, s in filtered[:6]]
        scores = [s for d, s in filtered[:6]]

        if not verified_docs:
            return {
                "success": True,
                "verdict": "unverifiable",
                "confidence": 0,
                "message": "No sufficiently relevant evidence found.",
                "evidence": []
            }

        # Compute a simple support score
        avg_score = sum(scores) / len(scores) if scores else 0
        if avg_score >= 0.78:
            verdict = "supported"
            confidence = min(10, int(avg_score * 12))
        elif avg_score >= 0.55:
            verdict = "partial"
            confidence = min(7, int(avg_score * 10))
        else:
            verdict = "contradicted_or_absent"
            confidence = max(1, int(avg_score * 5))

        evidence = []
        for doc, score in zip(verified_docs, scores):
            evidence.append({
                "content": doc.page_content[:600],
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "?"),
                "score": round(float(score), 4)
            })

        return {
            "success": True,
            "verdict": verdict,
            "confidence": confidence,
            "statement_checked": statement,
            "evidence": evidence,
            "message": f"Verification complete. Verdict: {verdict} (confidence: {confidence}/10)"
        }

    def _get_document_list(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute get_document_list tool"""
        if not hasattr(self.rag_pipeline, 'documents') or not self.rag_pipeline.documents:
            return {
                "success": False,
                "error": "No documents loaded in the knowledge base"
            }

        # Extract unique document names and metadata
        doc_info = {}
        for doc in self.rag_pipeline.documents:
            source = doc.metadata.get("source", "Unknown")
            if source not in doc_info:
                doc_info[source] = {
                    "name": source,
                    "pages": set(),
                    "types": set(),
                    "chunk_count": 0
                }

            # Use metadata safely
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            doc_info[source]["pages"].add(str(metadata.get("page", "?")))
            doc_info[source]["types"].add(str(metadata.get("type", "text")))
            doc_info[source]["chunk_count"] += 1

        # Format for output
        documents = []
        for source, info in doc_info.items():
            documents.append({
                "name": info["name"],
                "total_chunks": info["chunk_count"],
                "page_count": len(info["pages"]),
                "content_types": list(info["types"])
            })

        return {
            "success": True,
            "message": f"Found {len(documents)} documents in knowledge base",
            "documents": documents,
            "total_documents": len(documents)
        }

    def _plan_answer_structure(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute plan_answer_structure tool"""
        question = args.get("question", "")
        key_findings = args.get("key_findings", "")
        proposed_sections = args.get("proposed_sections", [])
        identified_intent = args.get("identified_intent", "Unknown")
        target_word_count = args.get("target_word_count", "Auto")

        # Create the structured plan summary
        structured_plan = []
        for i, section in enumerate(proposed_sections, 1):
            structured_plan.append(f"{i}. {section}")

        return {
            "success": True,
            "message": f"Answer structure plan verified for '{identified_intent}' intent.",
            "planned_sections": structured_plan,
            "intent": identified_intent,
            "target_length": target_word_count,
            "instruction": (
                f"You have committed to a **{identified_intent}** response with a target of **{target_word_count} words**. "
                "Use **TABLES** for all direct comparisons of penalties, zones, or numerical data. "
                "Use **BULLETS** with 2-3 sentence explanations for descriptive features. "
                "Proceed to write your Final Answer using the planned sections. Ensure 100% accuracy and verify all key facts."
            )
        }


def get_tool_schemas_for_gemini():
    """
    Convert tool schemas to Gemini API format

    Returns:
        List of tool declarations for Gemini API
    """
    return TOOL_SCHEMAS


def format_tool_result_for_prompt(tool_name: str, result: Dict[str, Any]) -> str:
    """
    Format tool execution result as a text prompt for the LLM

    Args:
        tool_name: Name of the executed tool
        result: Tool execution result

    Returns:
        Formatted string for inclusion in prompt
    """
    if not result.get("success", False):
        return f"[Tool Error - {tool_name}]: {result.get('error', 'Unknown error')}"

    if tool_name in ("retrieve_documents", "search_specific_document"):
        docs = result.get("documents", [])
        if not docs:
            return f"[{tool_name}]: No relevant documents found."

        formatted = f"[{tool_name}]: Retrieved {len(docs)} highly relevant chunks:\n\n"
        for i, doc in enumerate(docs, 1):
            source = doc['source']
            page = doc['page']
            doc_type = doc['type']
            relevance = doc.get('relevance', 'UNKNOWN')

            # Extract authority/year if present in source name
            year_match = re.search(r'\((\d{4})\)', source)
            year_info = f", Year: {year_match.group(1)}" if year_match else ""

            formatted += f"### SOURCE {i}: {source} (Page {page}){year_info}\n"
            formatted += f"Type: {doc_type} | Relevance: {relevance}\n"
            formatted += f"Content: {doc['content']}\n"
            formatted += f"--- End of Source {i} ---\n\n"

        return formatted

    elif tool_name == "get_document_list":
        docs = result.get("documents", [])
        formatted = f"[{tool_name}]: Available documents:\n\n"
        for doc in docs:
            formatted += f"- {doc['name']}: {doc['total_chunks']} chunks, {doc['page_count']} pages\n"
        return formatted

    elif tool_name == "verify_answer":
        verdict = result.get("verdict", "unknown")
        confidence = result.get("confidence", 0)
        statement = result.get("statement_checked", "")
        evidence = result.get("evidence", [])

        verdict_symbol = {"supported": "✅", "partial": "⚠️", "contradicted_or_absent": "❌", "unverifiable": "❓"}.get(verdict, "❓")
        formatted = (
            f"[verify_answer]: {verdict_symbol} Statement: \"{statement}\"\n"
            f"Verdict: {verdict.upper()} | Confidence: {confidence}/10\n\n"
        )
        if evidence:
            formatted += "Supporting Evidence:\n"
            for i, ev in enumerate(evidence, 1):
                formatted += f"  Evidence {i}: [{ev['source']}, Page {ev['page']}]\n"
                formatted += f"  Content: {ev['content'][:400]}\n"
                formatted += f"  --- End Evidence {i} ---\n\n"
        return formatted

    elif tool_name == "plan_answer_structure":
        sections = result.get("planned_sections", [])
        intent = result.get("intent", "Unknown")
        target_length = result.get("target_length", "Auto")
        instruction = result.get("instruction", "")
        
        formatted = f"[{tool_name}]: STRUCTURE PLAN APPROVED\n"
        formatted += f"Identified Intent: {intent}\n"
        formatted += f"Target Word Count: {target_length}\n"
        formatted += "Planned Sections:\n"
        for s in sections:
            formatted += f"- {s}\n"
        formatted += f"\nInstruction: {instruction}"
        return formatted

    return f"[{tool_name}]: {json.dumps(result, indent=2)}"
