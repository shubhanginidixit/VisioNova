"""
Input Classifier
Detects whether user input is a claim, question, or URL.
"""
import re
from urllib.parse import urlparse


class InputClassifier:
    """Classifies user input into different types."""
    
    # Patterns for questions
    QUESTION_STARTERS = [
        'is ', 'are ', 'was ', 'were ', 'do ', 'does ', 'did ',
        'can ', 'could ', 'will ', 'would ', 'should ',
        'has ', 'have ', 'had ', 'what ', 'when ', 'where ',
        'who ', 'why ', 'how ', 'which '
    ]
    
    def __init__(self):
        pass
    
    def classify(self, user_input: str) -> dict:
        """
        Classify the input and return its type with processed content.
        
        Args:
            user_input: Raw text from user (claim, question, or URL)
            
        Returns:
            dict with 'type', 'original', and 'claim' (normalized for searching)
        """
        user_input = user_input.strip()
        
        # Check if it's a URL
        if self._is_url(user_input):
            return {
                'type': 'url',
                'original': user_input,
                'claim': None  # Will be extracted from URL content
            }
        
        # Check if it's a question
        if self._is_question(user_input):
            claim = self._question_to_claim(user_input)
            return {
                'type': 'question',
                'original': user_input,
                'claim': claim
            }
        
        # Otherwise treat as a claim/statement
        return {
            'type': 'claim',
            'original': user_input,
            'claim': self._normalize_claim(user_input)
        }
    
    def _is_url(self, text: str) -> bool:
        """Check if text is a valid URL."""
        try:
            result = urlparse(text)
            return all([result.scheme in ('http', 'https'), result.netloc])
        except:
            return False
    
    def _is_question(self, text: str) -> bool:
        """Check if text is a question."""
        text_lower = text.lower().strip()
        
        # Check for question mark
        if text.endswith('?'):
            return True
        
        # Check for question starters
        for starter in self.QUESTION_STARTERS:
            if text_lower.startswith(starter):
                return True
        
        return False
    
    def _question_to_claim(self, question: str) -> str:
        """
        Convert a question to a searchable claim.
        E.g., "Is the Earth round?" -> "Earth is round"
        """
        # Remove question mark
        claim = question.rstrip('?').strip()
        
        # Simple transformations for common patterns
        lower = claim.lower()
        
        # "Is X Y?" -> "X is Y"
        if lower.startswith('is '):
            claim = claim[3:] + ' is'
        elif lower.startswith('are '):
            claim = claim[4:] + ' are'
        elif lower.startswith('was '):
            claim = claim[4:] + ' was'
        elif lower.startswith('were '):
            claim = claim[5:] + ' were'
        elif lower.startswith('did '):
            claim = claim[4:]
        elif lower.startswith('does '):
            claim = claim[5:]
        elif lower.startswith('do '):
            claim = claim[3:]
        
        return self._normalize_claim(claim)
    
    def _normalize_claim(self, claim: str) -> str:
        """
        Normalize a claim for searching.
        Strips metadata, timestamps, and cleans article-like text.
        """
        # Remove common article metadata patterns
        # Pattern: "Updated - January 16, 2026 09:14 am IST"
        claim = re.sub(
            r'^(Updated|Published|Posted|Modified)\s*[-–:]\s*\w+\s+\d{1,2},?\s+\d{4}\s+\d{1,2}:\d{2}\s*(am|pm|AM|PM)?\s*\w*\s*',
            '',
            claim,
            flags=re.IGNORECASE
        )
        
        # Remove standalone date patterns at the beginning
        # Pattern: "January 15, 2026" or "15 January 2026" or "2026-01-15"
        claim = re.sub(
            r'^\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\w+\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+\w+\s+\d{4})\s*[-–:]?\s*',
            '',
            claim,
            flags=re.IGNORECASE
        )
        
        # Remove time patterns like "09:14 am IST" or "10:30 PM EST"
        claim = re.sub(
            r'\d{1,2}:\d{2}\s*(am|pm|AM|PM)?\s*(IST|EST|PST|GMT|UTC|CST|MST)?\s*',
            '',
            claim
        )
        
        # Remove "Updated -", "Published:", etc. at start
        claim = re.sub(
            r'^(Updated|Published|Posted|Modified|Last updated|Source|Photo|Image|Video)\s*[-–:]\s*',
            '',
            claim,
            flags=re.IGNORECASE
        )
        
        # Remove source attributions like "(Reuters)" or "- AP News"
        claim = re.sub(
            r'\s*[-–]\s*(Reuters|AP|AFP|PTI|ANI|IANS|UNI)\s*$',
            '',
            claim,
            flags=re.IGNORECASE
        )
        claim = re.sub(
            r'\s*\((Reuters|AP|AFP|PTI|ANI|IANS|UNI)\)\s*$',
            '',
            claim,
            flags=re.IGNORECASE
        )
        
        # Remove extra whitespace
        claim = ' '.join(claim.split())
        
        # Remove leading/trailing punctuation (except important ones)
        claim = claim.strip('.,;:!?-–')
        
        # If the claim is too long (likely pasted article), truncate to first sentence
        if len(claim) > 200:
            # Find first sentence break
            sentence_end = re.search(r'[.!?]\s', claim)
            if sentence_end and sentence_end.start() > 30:
                claim = claim[:sentence_end.start() + 1]
        
        return claim.strip()



# Quick test
if __name__ == '__main__':
    classifier = InputClassifier()
    
    tests = [
        "The Eiffel Tower is 300 meters tall",
        "Is the moon landing real?",
        "https://example.com/article",
        "Did Einstein invent the lightbulb?",
        "COVID-19 vaccines contain microchips"
    ]
    
    for test in tests:
        result = classifier.classify(test)
        print(f"Input: {test}")
        print(f"  Type: {result['type']}")
        print(f"  Claim: {result['claim']}")
        print()
