import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ContextManager:
    """Manages context size for LLM interactions"""
    
    @staticmethod
    def optimize_system_context(context: str, max_length: int = 2000) -> str:
        """Optimize system context to fit within max_length"""
        if len(context) <= max_length:
            return context
        
        # Split into sections
        sections = context.split("\n\n")
        
        # If we have sections, optimize each
        if len(sections) > 1:
            # Start with crucial sections
            optimized = []
            remaining_length = max_length
            
            # Keep track of which sections we've added
            sections_added = set()
            
            # First prioritize anomalies section if it exists
            for i, section in enumerate(sections):
                if "ANOMALIE RECENTI" in section:
                    # Take top 5 anomalies
                    lines = section.split("\n")
                    # Keep header and up to 5 anomalies
                    anomaly_section = lines[0] + "\n" + "\n".join(lines[1:6])
                    if len(anomaly_section) <= remaining_length:
                        optimized.append(anomaly_section)
                        remaining_length -= len(anomaly_section)
                        sections_added.add(i)
                    break
            
            # Then add other sections if space allows
            for i, section in enumerate(sections):
                if i in sections_added:
                    continue  # Skip already added sections
                
                if len(section) <= remaining_length:
                    optimized.append(section)
                    remaining_length -= len(section)
                else:
                    # For long sections, truncate
                    if "MANUTENZIONI RECENTI" in section or "GUASTI RECENTI" in section:
                        lines = section.split("\n")
                        # Keep header and a few entries
                        truncated = lines[0] + "\n" + "\n".join(lines[1:3]) + "\n..."
                        if len(truncated) <= remaining_length:
                            optimized.append(truncated)
                            remaining_length -= len(truncated)
            
            # Join sections back
            return "\n\n".join(optimized)
        else:
            # Simple truncation with indicator
            return context[:max_length-3] + "..."

    @staticmethod
    def prioritize_query_context(query: str, context_docs: List[Dict], max_docs: int = 5) -> List[Dict]:
        """Prioritize which context documents to keep based on the query"""
        if len(context_docs) <= max_docs:
            return context_docs
            
        # Extract keywords from query
        query_lower = query.lower()
        
        # Define important keywords and their associated categories
        keywords = {
            "anomalia": "anomaly",
            "guasto": "failure", 
            "manutenzione": "maintenance",
            "errore": "anomaly",
            "problema": "failure",
            "energia": "energy",
            "consumo": "energy",
            "cos": "cosphi",
            "tensione": "voltage",
            "corrente": "current"
        }
        
        # Check which keywords are in the query
        matched_categories = []
        for keyword, category in keywords.items():
            if keyword in query_lower:
                matched_categories.append(category)
                
        # Score documents based on relevance to query categories
        scored_docs = []
        for doc in context_docs:
            score = 0
            doc_text = str(doc.get("content", "")).lower()
            
            # Score based on matched categories
            for category in matched_categories:
                if category in doc_text:
                    score += 3
                
            # Score based on recency if timestamp exists
            if "timestamp" in doc or "date" in doc:
                timestamp = doc.get("timestamp", doc.get("date"))
                if isinstance(timestamp, datetime):
                    # Higher score for more recent documents
                    days_old = (datetime.now() - timestamp).days
                    recency_score = max(0, 5 - days_old/7)  # Higher score for newer docs
                    score += recency_score
            
            scored_docs.append((score, doc))
            
        # Sort by score and take top max_docs
        scored_docs.sort(reverse=True)
        return [doc for _, doc in scored_docs[:max_docs]]

    @staticmethod
    def optimize_chat_history(history: List[Dict], max_turns: int = 3) -> List[Dict]:
        """Keep only the most recent conversation turns"""
        if len(history) <= max_turns * 2:  # Each turn has user + assistant message
            return history
            
        # Keep the most recent turns
        return history[-(max_turns * 2):]
