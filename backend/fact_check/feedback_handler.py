"""
Feedback Handler
Manages user feedback on fact-check verdicts.
"""
import json
import os
import hashlib
from datetime import datetime


class FeedbackHandler:
    """Handles user feedback on fact-checking results."""
    
    def __init__(self, feedback_file: str = None):
        """
        Initialize feedback handler.
        
        Args:
            feedback_file: Path to JSON file storing feedback (defaults to same directory)
        """
        if feedback_file is None:
            feedback_file = os.path.join(
                os.path.dirname(__file__),
                'user_feedback.json'
            )
        
        self.feedback_file = feedback_file
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create feedback file if it doesn't exist."""
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
    
    def _load_feedback(self) -> list:
        """Load all feedback from file."""
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_feedback(self, feedback_list: list):
        """Save feedback list to file."""
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_list, f, indent=2, ensure_ascii=False)
    
    def submit_feedback(self, claim: str, original_verdict: str, 
                       user_verdict: str, reason: str = None, 
                       additional_sources: list = None) -> dict:
        """
        Submit user feedback on a fact-check result.
        
        Args:
            claim: The original claim that was fact-checked
            original_verdict: Verdict given by the system
            user_verdict: User's opinion on the verdict
            reason: User's explanation (optional)
            additional_sources: Additional source URLs provided by user (optional)
            
        Returns:
            dict with success status and feedback_id
        """
        # Generate unique ID for this feedback
        feedback_id = hashlib.md5(
            f"{claim}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        feedback_entry = {
            'feedback_id': feedback_id,
            'timestamp': datetime.now().isoformat(),
            'claim': claim,
            'original_verdict': original_verdict,
            'user_verdict': user_verdict,
            'reason': reason or '',
            'additional_sources': additional_sources or [],
            'status': 'pending_review'
        }
        
        # Load existing feedback
        all_feedback = self._load_feedback()
        
        # Add new feedback
        all_feedback.append(feedback_entry)
        
        # Save updated feedback
        self._save_feedback(all_feedback)
        
        return {
            'success': True,
            'feedback_id': feedback_id,
            'message': 'Feedback submitted successfully'
        }
    
    def get_feedback_stats(self) -> dict:
        """Get statistics about submitted feedback."""
        all_feedback = self._load_feedback()
        
        total = len(all_feedback)
        pending = sum(1 for f in all_feedback if f.get('status') == 'pending_review')
        reviewed = total - pending
        
        # Count disagreements by verdict
        verdict_disagreements = {}
        for f in all_feedback:
            orig = f.get('original_verdict', 'UNKNOWN')
            if orig not in verdict_disagreements:
                verdict_disagreements[orig] = 0
            verdict_disagreements[orig] += 1
        
        return {
            'total_feedback': total,
            'pending_review': pending,
            'reviewed': reviewed,
            'disagreements_by_verdict': verdict_disagreements
        }
    
    def get_recent_feedback(self, limit: int = 10) -> list:
        """Get most recent feedback entries."""
        all_feedback = self._load_feedback()
        # Sort by timestamp descending
        all_feedback.sort(
            key=lambda x: x.get('timestamp', ''), 
            reverse=True
        )
        return all_feedback[:limit]


# Quick test
if __name__ == '__main__':
    handler = FeedbackHandler()
    
    # Submit test feedback
    result = handler.submit_feedback(
        claim="The moon landing was fake",
        original_verdict="FALSE",
        user_verdict="TRUE",
        reason="I have evidence this was staged",
        additional_sources=["https://example.com/proof"]
    )
    
    print(f"Feedback submitted: {result}")
    print(f"\nStats: {handler.get_feedback_stats()}")
    print(f"\nRecent feedback: {handler.get_recent_feedback(limit=5)}")
