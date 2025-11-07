"""
Enrich chunk metadata with better topic/section information.

This script re-processes chunks to add better topic classification,
improve heading paths, and add semantic tags.

Usage: python scripts/enrich_chunks.py
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any


def classify_content_type(content: str, heading: str) -> str:
    """Classify what type of content this is."""
    content_lower = content.lower() if content else ''
    heading_lower = heading.lower() if heading else ''

    # Integration guides
    if any(word in content_lower or word in heading_lower
           for word in ['integration', 'integrate', 'connect', 'api', 'webhook']):
        return 'integration'

    # Setup/configuration
    if any(word in content_lower or word in heading_lower
           for word in ['setup', 'configure', 'settings', 'install']):
        return 'configuration'

    # How-to guides
    if any(word in content_lower or word in heading_lower
           for word in ['how to', 'step', 'guide', 'tutorial']):
        return 'procedural'

    # Troubleshooting
    if any(word in content_lower or word in heading_lower
           for word in ['troubleshoot', 'error', 'issue', 'problem', 'fix']):
        return 'troubleshooting'

    # Feature descriptions
    if any(word in content_lower or word in heading_lower
           for word in ['feature', 'functionality', 'capability', 'about']):
        return 'conceptual'

    # Security/permissions
    if any(word in content_lower or word in heading_lower
           for word in ['security', 'permission', 'access', 'role', 'authentication']):
        return 'security'

    return 'general'


def extract_integration_names(content: str, heading: str) -> List[str]:
    """Extract integration names from content."""
    integrations = []

    # Common integrations
    integration_patterns = [
        'MS Teams', 'Microsoft Teams', 'Slack', 'WhatsApp', 'Shopify',
        'Facebook Messenger', 'Instagram', 'Twitter', 'LinkedIn',
        'Zapier', 'Make', 'Google Analytics', 'Salesforce'
    ]

    content = content or ''
    heading = heading or ''
    combined = content + ' ' + heading

    for integration in integration_patterns:
        if integration.lower() in combined.lower():
            integrations.append(integration)

    return list(set(integrations))


def extract_key_topics(content: str, heading: str) -> List[str]:
    """Extract key topics/concepts from content."""
    topics = []

    # Topic patterns
    topic_patterns = {
        'chatbot': ['chatbot', 'bot', 'conversation'],
        'automation': ['automation', 'workflow', 'trigger', 'automate'],
        'testing': ['test', 'testing', 'qa', 'quality assurance'],
        'analytics': ['analytics', 'report', 'metrics', 'dashboard'],
        'channels': ['channel', 'messaging', 'communication'],
        'users': ['user', 'contact', 'customer', 'visitor'],
        'templates': ['template', 'message template', 'response'],
        'ai': ['ai', 'artificial intelligence', 'nlp', 'machine learning'],
        'api': ['api', 'endpoint', 'rest', 'webhook'],
        'security': ['security', 'permission', 'access control', 'authentication']
    }

    content = content or ''
    heading = heading or ''
    combined_lower = (content + ' ' + heading).lower()

    for topic, patterns in topic_patterns.items():
        if any(pattern in combined_lower for pattern in patterns):
            topics.append(topic)

    return topics


def improve_heading_path(heading_path: List[str], content: str) -> List[str]:
    """Improve heading path by inferring missing levels."""
    if not heading_path or len(heading_path) < 2:
        # Try to infer from content
        content = content or ''
        content_lower = content.lower()

        # Common top-level sections
        if 'getting started' in content_lower or 'introduction' in content_lower:
            return ['Getting Started'] + heading_path
        elif 'integration' in content_lower:
            return ['Integrations'] + heading_path
        elif 'feature' in content_lower or 'functionality' in content_lower:
            return ['Features'] + heading_path
        elif 'setting' in content_lower or 'configuration' in content_lower:
            return ['Configuration'] + heading_path

    return heading_path


def enrich_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich a single chunk with better metadata."""
    content = chunk.get('content', '')
    metadata = chunk.get('metadata', {})

    heading = metadata.get('current_heading', '')
    heading_path = metadata.get('heading_path', [])

    # Classify content type
    content_type = classify_content_type(content, heading)
    metadata['content_type'] = content_type

    # Extract integrations
    integrations = extract_integration_names(content, heading)
    if integrations:
        metadata['integration_names'] = integrations

    # Extract topics
    topics = extract_key_topics(content, heading)
    if topics:
        metadata['topics'] = topics

    # Improve heading path
    improved_path = improve_heading_path(heading_path, content)
    if improved_path != heading_path:
        metadata['heading_path_improved'] = improved_path

    # Add searchable summary (first 200 chars)
    metadata['content_summary'] = content[:200].strip()

    # Technical depth estimation
    technical_keywords = ['api', 'code', 'endpoint', 'json', 'authentication', 'token', 'developer']
    technical_count = sum(1 for kw in technical_keywords if kw in content.lower())
    metadata['technical_depth'] = 'high' if technical_count >= 3 else 'medium' if technical_count >= 1 else 'low'

    chunk['metadata'] = metadata
    return chunk


def main():
    """Main enrichment process."""
    print("🔧 Starting chunk enrichment...")

    # Load chunks
    chunks_path = Path('cache/hierarchical_chunks_filtered.json')
    if not chunks_path.exists():
        print(f"❌ Error: {chunks_path} not found!")
        return

    with open(chunks_path) as f:
        data = json.load(f)

    # Handle both list and dict structures
    if isinstance(data, dict) and 'chunks' in data:
        chunks = data['chunks']
        original_structure = data
    else:
        chunks = data
        original_structure = None

    print(f"📦 Loaded {len(chunks)} chunks")

    # Enrich each chunk
    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        if i % 100 == 0:
            print(f"  Processing chunk {i}/{len(chunks)}...")
        enriched_chunk = enrich_chunk(chunk)
        enriched_chunks.append(enriched_chunk)

    # Save enriched chunks
    output_path = Path('cache/hierarchical_chunks_enriched.json')

    if original_structure:
        # Preserve original structure
        original_structure['chunks'] = enriched_chunks
        with open(output_path, 'w') as f:
            json.dump(original_structure, f, indent=2)
    else:
        with open(output_path, 'w') as f:
            json.dump(enriched_chunks, f, indent=2)

    print(f"\n✅ Enriched chunks saved to: {output_path}")

    # Show sample
    print("\n📊 Sample enriched chunk:")
    sample = enriched_chunks[100]
    print(f"  Content type: {sample['metadata'].get('content_type', 'N/A')}")
    print(f"  Topics: {sample['metadata'].get('topics', [])}")
    print(f"  Integrations: {sample['metadata'].get('integration_names', [])}")
    print(f"  Technical depth: {sample['metadata'].get('technical_depth', 'N/A')}")

    print("\n💡 Next steps:")
    print("  1. Review the enriched chunks in cache/hierarchical_chunks_enriched.json")
    print("  2. If satisfied, replace hierarchical_chunks_filtered.json with enriched version")
    print("  3. Re-run: python -m src.database.run_phase5")
    print("  4. Test with: python scripts/diagnose_quality.py \"your test query\"")


if __name__ == "__main__":
    main()
