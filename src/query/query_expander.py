"""
Query Expansion Module.

Expands queries with synonyms, related terms, and integration name variations
to improve retrieval recall and precision.
"""

import logging
from typing import List, Dict, Set
import re

logger = logging.getLogger(__name__)


class QueryExpander:
    """
    Expands queries with domain-specific synonyms and variations.

    This helps retrieve relevant chunks even when users use different terminology
    than what appears in the documentation.
    """

    def __init__(self):
        """Initialize query expander with domain-specific mappings."""

        # Action/verb synonyms (common in procedural queries)
        self.action_synonyms = {
            'integrate': ['connect', 'link', 'setup', 'configure', 'add', 'enable'],
            'create': ['make', 'build', 'add', 'generate', 'set up', 'initialize'],
            'configure': ['setup', 'set up', 'customize', 'adjust', 'modify'],
            'install': ['setup', 'set up', 'deploy', 'add'],
            'test': ['verify', 'validate', 'check', 'run tests', 'execute tests'],
            'debug': ['troubleshoot', 'fix', 'diagnose', 'solve'],
            'automate': ['script', 'automated', 'automation'],
            'run': ['execute', 'start', 'launch', 'trigger'],
            'import': ['load', 'upload', 'bring in', 'add'],
            'export': ['download', 'save', 'extract'],
        }

        # Integration name variations (critical for integration queries)
        self.integration_aliases = {
            'ms teams': ['microsoft teams', 'teams'],
            'microsoft teams': ['ms teams', 'teams'],
            'slack': ['slack messenger', 'slack workspace'],
            'whatsapp': ['whatsapp business', 'wa'],
            'shopify': ['shopify store', 'shopify ecommerce'],
            'jira': ['atlassian jira', 'jira software'],
            'azure': ['microsoft azure', 'azure devops'],
            'aws': ['amazon web services', 'amazon aws'],
            'salesforce': ['sfdc', 'salesforce crm'],
        }

        # Technical term synonyms
        self.technical_synonyms = {
            'api': ['rest api', 'web api', 'endpoint', 'web service'],
            'webhook': ['web hook', 'callback', 'http callback'],
            'authentication': ['auth', 'login', 'credentials', 'sign in'],
            'authorization': ['permissions', 'access control', 'privileges'],
            'database': ['db', 'data store', 'data source'],
            'error': ['issue', 'problem', 'bug', 'failure'],
            'no-code': ['no code', 'nocode', 'low-code', 'visual programming'],
            'chatbot': ['bot', 'conversational ai', 'virtual assistant'],
        }

        # Concept expansions (add related terms)
        self.concept_expansions = {
            'testing': ['qa', 'quality assurance', 'test automation', 'test cases'],
            'security': ['permissions', 'access control', 'authentication', 'encryption'],
            'automation': ['workflow', 'automated', 'automate', 'scripting'],
            'reporting': ['analytics', 'dashboards', 'metrics', 'insights'],
            'integration': ['connector', 'connection', 'sync', 'bridge'],
        }

        # Combine all synonym mappings
        self.all_synonyms = {
            **self.action_synonyms,
            **self.integration_aliases,
            **self.technical_synonyms,
            **self.concept_expansions
        }

        logger.info(f"Initialized QueryExpander with {len(self.all_synonyms)} synonym mappings")

    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """
        Expand query with synonyms and variations.

        Args:
            query: Original query string
            max_expansions: Maximum number of expanded variations to return

        Returns:
            List of query variations (always includes original as first item)
        """
        variations = [query]  # Always include original query first
        query_lower = query.lower()

        # Find matching terms in the query
        matched_terms = []
        for term, synonyms in self.all_synonyms.items():
            # Check if term appears in query (with word boundaries)
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, query_lower):
                matched_terms.append((term, synonyms))

        # Generate variations by replacing terms with synonyms
        if matched_terms:
            # Sort by term length (longer terms first to avoid partial matches)
            matched_terms.sort(key=lambda x: len(x[0]), reverse=True)

            # For each matched term, create variations with its top synonyms
            for term, synonyms in matched_terms[:2]:  # Limit to top 2 terms to avoid explosion
                for synonym in synonyms[:2]:  # Use top 2 synonyms per term
                    # Replace term with synonym (case-insensitive)
                    pattern = r'\b' + re.escape(term) + r'\b'
                    expanded = re.sub(pattern, synonym, query_lower)

                    if expanded != query_lower and expanded not in [v.lower() for v in variations]:
                        variations.append(expanded)

                    # Stop if we have enough variations
                    if len(variations) >= max_expansions + 1:  # +1 for original
                        break

                if len(variations) >= max_expansions + 1:
                    break

        # Limit to max_expansions + original
        variations = variations[:max_expansions + 1]

        if len(variations) > 1:
            logger.debug(f"Expanded query into {len(variations)} variations")
            for i, var in enumerate(variations):
                logger.debug(f"  {i+1}. {var}")

        return variations

    def expand_for_integration(self, query: str, integration_name: str) -> List[str]:
        """
        Specifically expand for integration queries.

        Args:
            query: Original query
            integration_name: Name of the integration (e.g., "MS Teams")

        Returns:
            List of query variations with integration name variations
        """
        variations = [query]

        # Get aliases for this integration
        integration_lower = integration_name.lower()
        aliases = self.integration_aliases.get(integration_lower, [])

        if aliases:
            for alias in aliases:
                # Replace integration name with alias
                pattern = re.compile(re.escape(integration_name), re.IGNORECASE)
                expanded = pattern.sub(alias, query)
                if expanded.lower() != query.lower():
                    variations.append(expanded)

        return variations[:4]  # Limit to 4 variations

    def get_search_keywords(self, query: str) -> Set[str]:
        """
        Extract key searchable terms from query including synonyms.

        Useful for keyword boosting in retrieval.

        Args:
            query: Query string

        Returns:
            Set of important keywords including synonyms
        """
        keywords = set()
        query_lower = query.lower()

        # Add original words (excluding common stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                     'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
                     'how', 'what', 'when', 'where', 'why', 'do', 'does', 'i'}

        words = re.findall(r'\b\w+\b', query_lower)
        for word in words:
            if word not in stop_words and len(word) > 2:
                keywords.add(word)

        # Add synonyms for matched terms
        for term, synonyms in self.all_synonyms.items():
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, query_lower):
                keywords.add(term)
                # Add first 2 synonyms
                keywords.update(synonyms[:2])

        return keywords


def main():
    """Test the query expander."""
    expander = QueryExpander()

    print("="*80)
    print("🔍 QUERY EXPANDER TEST")
    print("="*80)

    test_queries = [
        "How do I integrate MS Teams with Watermelon?",
        "How do I create a no-code block?",
        "What are the authentication options?",
        "How do I test my chatbot?",
        "How do I configure Slack integration?",
    ]

    for query in test_queries:
        print(f"\n📝 Original: {query}")
        print("-"*80)

        expansions = expander.expand_query(query, max_expansions=3)
        print(f"✨ Expanded to {len(expansions)} variations:")
        for i, expansion in enumerate(expansions, 1):
            marker = "  [ORIGINAL]" if i == 1 else ""
            print(f"  {i}. {expansion}{marker}")

        keywords = expander.get_search_keywords(query)
        print(f"\n🔑 Key terms ({len(keywords)}): {', '.join(sorted(keywords)[:10])}")

    print("\n" + "="*80)
    print("✅ Query expander test complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
