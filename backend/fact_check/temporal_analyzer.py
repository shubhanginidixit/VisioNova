"""
Temporal Analyzer
Extracts dates, years, and temporal context from claims and URLs.
"""
import re
from datetime import datetime
from typing import Optional, Dict


class TemporalAnalyzer:
    """Analyzes temporal context in claims and content."""
    
    # Month names for extraction
    MONTHS = [
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        'jan', 'feb', 'mar', 'apr', 'may', 'jun',
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ]
    
    def __init__(self):
        self.current_year = datetime.now().year
    
    def extract_temporal_context(self, text: str, url: str = None) -> Dict:
        """
        Extract temporal context from text and URL.
        
        Args:
            text: The claim or article text
            url: Optional URL of the article
            
        Returns:
            dict with year_mentioned, date_mentioned, temporal_keywords, search_year_from
        """
        # Extract years
        years = self._extract_years(text)
        
        # Extract full dates
        dates = self._extract_dates(text)
        
        # Check for temporal keywords
        temporal_keywords = self._extract_temporal_keywords(text)
        
        # Determine the most relevant year for searching
        search_year_from = self._determine_search_year(years, dates, url)
        
        return {
            'years_mentioned': years,
            'dates_mentioned': dates,
            'temporal_keywords': temporal_keywords,
            'search_year_from': search_year_from,
            'is_historical': search_year_from and search_year_from < (self.current_year - 5),
            'is_recent': search_year_from and search_year_from >= (self.current_year - 2),
            'time_period': self._categorize_period(search_year_from)
        }
    
    def _extract_years(self, text: str) -> list:
        """Extract 4-digit years from text."""
        # Match 4-digit years (1900-2099)
        year_pattern = r'\b(19\d{2}|20\d{2})\b'
        years = re.findall(year_pattern, text)
        return sorted([int(y) for y in set(years)])
    
    def _extract_dates(self, text: str) -> list:
        """Extract full dates from text."""
        dates = []
        
        # Pattern: "January 15, 2020" or "15 January 2020"
        date_patterns = [
            r'(\w+)\s+(\d{1,2}),?\s+(19\d{2}|20\d{2})',
            r'(\d{1,2})\s+(\w+)\s+(19\d{2}|20\d{2})',
            r'(19\d{2}|20\d{2})-(\d{1,2})-(\d{1,2})',
            r'(\d{1,2})/(\d{1,2})/(19\d{2}|20\d{2})'
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                dates.append(match.group(0))
        
        return dates
    
    def _extract_temporal_keywords(self, text: str) -> list:
        """Extract temporal keywords indicating time period."""
        keywords = []
        text_lower = text.lower()
        
        temporal_terms = {
            'historical': ['historical', 'history', 'ancient', 'medieval', 'century ago'],
            'recent': ['recent', 'recently', 'latest', 'new', 'current', 'today', 'this year'],
            'past': ['past', 'previous', 'former', 'earlier', 'ago'],
            'ongoing': ['ongoing', 'continuing', 'still', 'currently'],
            'decade': ['1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s']
        }
        
        for category, terms in temporal_terms.items():
            for term in terms:
                if term in text_lower:
                    keywords.append({'category': category, 'term': term})
        
        return keywords
    
    def _determine_search_year(self, years: list, dates: list, url: str = None) -> Optional[int]:
        """
        Determine the most relevant year to start archive searches from.
        
        Priority:
        1. Most recent year mentioned in text
        2. Year from URL if available
        3. Current year if no year found
        """
        # If years are mentioned, use the earliest (to get context from that time)
        if years:
            # For historical claims, use the earliest year mentioned
            # For recent claims, use the most recent year
            earliest_year = min(years)
            if earliest_year < (self.current_year - 5):
                return earliest_year  # Historical claim
            else:
                return max(years)  # Recent claim
        
        # Try to extract year from URL
        if url:
            url_years = self._extract_years(url)
            if url_years:
                return max(url_years)
        
        # Default to current year
        return self.current_year
    
    def _categorize_period(self, year: Optional[int]) -> str:
        """Categorize the time period."""
        if not year:
            return 'contemporary'
        
        age = self.current_year - year
        
        if age <= 2:
            return 'current'
        elif age <= 5:
            return 'recent'
        elif age <= 10:
            return 'past_decade'
        elif age <= 20:
            return 'past_two_decades'
        elif age <= 50:
            return 'historical_modern'
        else:
            return 'historical'
    
    def format_search_period_description(self, temporal_context: dict) -> str:
        """Generate human-readable description of search period."""
        year = temporal_context['search_year_from']
        period = temporal_context['time_period']
        
        if not year:
            return "contemporary sources"
        
        if period == 'current':
            return f"latest sources from {year} onwards"
        elif period == 'recent':
            return f"recent sources from {year} onwards"
        elif period == 'past_decade':
            return f"archives from {year} onwards"
        elif period in ['past_two_decades', 'historical_modern']:
            return f"historical archives from {year} onwards"
        else:
            return f"historical archives from {year} onwards"


# Quick test
if __name__ == '__main__':
    analyzer = TemporalAnalyzer()
    
    test_cases = [
        "The moon landing happened in 1969",
        "COVID-19 pandemic started in 2020",
        "Recent studies from 2024 show climate change effects",
        "The Berlin Wall fell in November 1989",
        "Latest news from today about AI developments"
    ]
    
    for claim in test_cases:
        context = analyzer.extract_temporal_context(claim)
        description = analyzer.format_search_period_description(context)
        
        print(f"\nClaim: {claim}")
        print(f"Search from: {context['search_year_from']}")
        print(f"Period: {context['time_period']}")
        print(f"Description: {description}")
        print(f"Is historical: {context['is_historical']}")
