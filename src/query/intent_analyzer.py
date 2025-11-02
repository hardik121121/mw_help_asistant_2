"""
Intent Analysis System.
Extracts user intent, goals, and entities from queries.
"""

import logging
import re
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class UserIntent(str, Enum):
    """High-level user intents."""
    LEARN = "learn"  # Understand a concept
    CREATE = "create"  # Create something new
    CONFIGURE = "configure"  # Set up or customize
    INTEGRATE = "integrate"  # Connect systems
    TROUBLESHOOT = "troubleshoot"  # Fix a problem
    COMPARE = "compare"  # Evaluate options
    OPTIMIZE = "optimize"  # Improve performance
    SECURE = "secure"  # Security/compliance
    MIGRATE = "migrate"  # Move/upgrade systems


class ActionType(str, Enum):
    """Specific actions user wants to take."""
    READ = "read"  # Read/view information
    WRITE = "write"  # Create/modify data
    DELETE = "delete"  # Remove something
    CONNECT = "connect"  # Establish connection
    CONFIGURE = "configure"  # Change settings
    TEST = "test"  # Verify functionality
    DEPLOY = "deploy"  # Release to production
    DEBUG = "debug"  # Find and fix issues


@dataclass
class Entity:
    """Named entity extracted from query."""
    text: str  # The entity text
    type: str  # Entity type (product, feature, technology, etc.)
    confidence: float = 0.8


@dataclass
class IntentAnalysis:
    """Result of intent analysis."""
    primary_intent: UserIntent
    secondary_intents: List[UserIntent] = field(default_factory=list)
    actions: List[ActionType] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    user_goal: str = ""  # Natural language description of goal
    prerequisites: List[str] = field(default_factory=list)  # What user needs first
    expected_outcome: str = ""  # What success looks like
    confidence: float = 0.8


class IntentAnalyzer:
    """
    Analyzes user intent and extracts actionable information from queries.

    Uses rule-based NLP techniques for fast, accurate intent extraction.
    """

    def __init__(self):
        """Initialize intent analyzer."""
        logger.info("Initialized IntentAnalyzer")

        # Intent keyword mappings
        self.intent_patterns = {
            UserIntent.LEARN: [
                "what is", "what are", "explain", "understand", "learn about",
                "tell me about", "describe", "definition", "overview"
            ],
            UserIntent.CREATE: [
                "create", "build", "make", "generate", "set up", "establish",
                "add", "new", "setup"
            ],
            UserIntent.CONFIGURE: [
                "configure", "customize", "set", "change", "modify", "adjust",
                "settings", "options", "preferences"
            ],
            UserIntent.INTEGRATE: [
                "integrate", "connect", "link", "sync", "api", "webhook",
                "third-party", "external"
            ],
            UserIntent.TROUBLESHOOT: [
                "fix", "solve", "error", "issue", "problem", "not working",
                "fails", "debug", "troubleshoot", "resolve"
            ],
            UserIntent.COMPARE: [
                "compare", "versus", "vs", "difference", "which", "better",
                "pros and cons", "advantages"
            ],
            UserIntent.OPTIMIZE: [
                "optimize", "improve", "performance", "faster", "efficient",
                "speed up", "enhance"
            ],
            UserIntent.SECURE: [
                "security", "secure", "protect", "permission", "access",
                "authentication", "authorization", "encrypt"
            ],
            UserIntent.MIGRATE: [
                "migrate", "upgrade", "move", "transfer", "switch", "transition"
            ]
        }

        # Action keyword mappings
        self.action_patterns = {
            ActionType.READ: ["view", "see", "read", "check", "find", "get"],
            ActionType.WRITE: ["create", "add", "write", "update", "modify", "edit"],
            ActionType.DELETE: ["delete", "remove", "clear", "erase"],
            ActionType.CONNECT: ["connect", "link", "join", "attach"],
            ActionType.CONFIGURE: ["configure", "set", "setup", "adjust"],
            ActionType.TEST: ["test", "verify", "validate", "check"],
            ActionType.DEPLOY: ["deploy", "release", "publish", "launch"],
            ActionType.DEBUG: ["debug", "troubleshoot", "diagnose", "fix"]
        }

        # Known entities (products, features, technologies)
        self.known_entities = {
            "products": ["watermelon", "ms teams", "slack", "salesforce", "jira"],
            "features": ["no-code block", "automation", "webhook", "api",
                        "autonomous testing", "workflow", "integration"],
            "technologies": ["rest api", "oauth", "json", "xml", "javascript",
                           "python", "webhook", "http"]
        }

    def analyze(self, query: str) -> IntentAnalysis:
        """
        Analyze user intent from query.

        Args:
            query: User's query string

        Returns:
            IntentAnalysis with intent, actions, entities, and goals
        """
        logger.info(f"Analyzing intent for: {query}")

        query_lower = query.lower()

        # Extract primary and secondary intents
        primary_intent, secondary_intents = self._extract_intents(query_lower)

        # Extract actions
        actions = self._extract_actions(query_lower)

        # Extract entities
        entities = self._extract_entities(query)

        # Determine user goal
        user_goal = self._determine_goal(query, primary_intent, actions)

        # Identify prerequisites
        prerequisites = self._identify_prerequisites(primary_intent, entities)

        # Determine expected outcome
        expected_outcome = self._determine_outcome(primary_intent, actions)

        analysis = IntentAnalysis(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            actions=actions,
            entities=entities,
            user_goal=user_goal,
            prerequisites=prerequisites,
            expected_outcome=expected_outcome,
            confidence=0.85
        )

        logger.info(f"Primary intent: {primary_intent.value}, Actions: {len(actions)}, "
                   f"Entities: {len(entities)}")
        return analysis

    def _extract_intents(self, query: str) -> tuple[UserIntent, List[UserIntent]]:
        """Extract primary and secondary intents."""
        intent_scores = {}

        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query)
            if score > 0:
                intent_scores[intent] = score

        if not intent_scores:
            # Default to LEARN if no clear intent
            return UserIntent.LEARN, []

        # Sort by score
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)

        primary = sorted_intents[0][0]
        secondary = [intent for intent, _ in sorted_intents[1:3]]  # Top 2 secondary

        return primary, secondary

    def _extract_actions(self, query: str) -> List[ActionType]:
        """Extract specific actions from query."""
        actions = []

        for action, patterns in self.action_patterns.items():
            if any(pattern in query for pattern in patterns):
                actions.append(action)

        return actions

    def _extract_entities(self, query: str) -> List[Entity]:
        """Extract named entities from query."""
        entities = []
        query_lower = query.lower()

        # Check known entities
        for entity_type, entity_list in self.known_entities.items():
            for entity_text in entity_list:
                if entity_text in query_lower:
                    entities.append(Entity(
                        text=entity_text,
                        type=entity_type[:-1],  # Remove plural 's'
                        confidence=0.9
                    ))

        # Extract capitalized phrases (likely product/feature names)
        words = query.split()
        current_phrase = []

        for word in words:
            clean = word.strip(",.?!:;")
            if clean and len(clean) > 2 and clean[0].isupper():
                current_phrase.append(clean)
            else:
                if len(current_phrase) >= 2:
                    phrase = " ".join(current_phrase)
                    # Avoid duplicates
                    if not any(e.text.lower() == phrase.lower() for e in entities):
                        entities.append(Entity(
                            text=phrase,
                            type="unknown",
                            confidence=0.7
                        ))
                current_phrase = []

        if len(current_phrase) >= 2:
            phrase = " ".join(current_phrase)
            if not any(e.text.lower() == phrase.lower() for e in entities):
                entities.append(Entity(
                    text=phrase,
                    type="unknown",
                    confidence=0.7
                ))

        # Deduplicate
        seen = set()
        unique_entities = []
        for e in entities:
            key = e.text.lower()
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)

        return unique_entities

    def _determine_goal(self, query: str, intent: UserIntent,
                       actions: List[ActionType]) -> str:
        """Determine user's high-level goal."""
        goals = {
            UserIntent.LEARN: f"Understand {self._extract_main_topic(query)}",
            UserIntent.CREATE: f"Create {self._extract_main_topic(query)}",
            UserIntent.CONFIGURE: f"Configure {self._extract_main_topic(query)}",
            UserIntent.INTEGRATE: f"Integrate {self._extract_main_topic(query)}",
            UserIntent.TROUBLESHOOT: f"Fix issues with {self._extract_main_topic(query)}",
            UserIntent.COMPARE: f"Compare options for {self._extract_main_topic(query)}",
            UserIntent.OPTIMIZE: f"Optimize {self._extract_main_topic(query)}",
            UserIntent.SECURE: f"Secure {self._extract_main_topic(query)}",
            UserIntent.MIGRATE: f"Migrate {self._extract_main_topic(query)}"
        }

        return goals.get(intent, "Answer user query")

    def _extract_main_topic(self, query: str) -> str:
        """Extract the main topic from query."""
        # Remove common question words
        topic = query.lower()
        remove_words = ["how to", "how do i", "what is", "what are", "can i",
                       "should i", "where", "when", "why"]

        for word in remove_words:
            topic = topic.replace(word, "")

        topic = topic.strip("?.,!: ")

        # Take first few meaningful words
        words = topic.split()[:5]
        return " ".join(words) if words else "the system"

    def _identify_prerequisites(self, intent: UserIntent,
                               entities: List[Entity]) -> List[str]:
        """Identify what user needs before achieving goal."""
        prerequisites = []

        if intent == UserIntent.CREATE:
            prerequisites.append("Understanding of basic concepts")
            prerequisites.append("Required permissions")

        elif intent == UserIntent.INTEGRATE:
            prerequisites.append("API credentials")
            prerequisites.append("Both systems accessible")

        elif intent == UserIntent.CONFIGURE:
            prerequisites.append("Access to settings")
            prerequisites.append("Understanding of options")

        elif intent == UserIntent.TROUBLESHOOT:
            prerequisites.append("Error details or logs")
            prerequisites.append("Current configuration")

        # Add entity-specific prerequisites
        for entity in entities:
            if "api" in entity.text.lower():
                prerequisites.append("API key or token")

        return list(set(prerequisites))  # Deduplicate

    def _determine_outcome(self, intent: UserIntent,
                          actions: List[ActionType]) -> str:
        """Determine expected successful outcome."""
        outcomes = {
            UserIntent.LEARN: "Clear understanding of the concept",
            UserIntent.CREATE: "Successfully created and functional",
            UserIntent.CONFIGURE: "Configured correctly and working as expected",
            UserIntent.INTEGRATE: "Systems connected and communicating",
            UserIntent.TROUBLESHOOT: "Issue resolved and system working",
            UserIntent.COMPARE: "Informed decision on best option",
            UserIntent.OPTIMIZE: "Improved performance and efficiency",
            UserIntent.SECURE: "Enhanced security posture",
            UserIntent.MIGRATE: "Successful migration with no data loss"
        }

        return outcomes.get(intent, "Query answered successfully")

    def to_dict(self, analysis: IntentAnalysis) -> Dict:
        """Convert IntentAnalysis to dictionary."""
        return asdict(analysis)


def main():
    """Test intent analyzer with sample queries."""
    print("\n" + "="*60)
    print("🎯 INTENT ANALYZER TEST")
    print("="*60 + "\n")

    analyzer = IntentAnalyzer()

    # Test queries
    test_queries = [
        "What is MS Teams integration?",
        "How do I create a no-code block on Watermelon and process it for Autonomous Functional Testing?",
        "Fix authentication error in Slack integration",
        "Compare Watermelon automation vs custom coding"
    ]

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 60)

        result = analyzer.analyze(query)

        print(f"Primary Intent: {result.primary_intent.value}")
        if result.secondary_intents:
            print(f"Secondary Intents: {', '.join(i.value for i in result.secondary_intents)}")

        print(f"Actions: {', '.join(a.value for a in result.actions)}")

        print(f"\nEntities ({len(result.entities)}):")
        for entity in result.entities:
            print(f"  - {entity.text} ({entity.type}) [confidence: {entity.confidence:.2f}]")

        print(f"\nUser Goal: {result.user_goal}")
        print(f"Expected Outcome: {result.expected_outcome}")

        if result.prerequisites:
            print(f"\nPrerequisites:")
            for prereq in result.prerequisites:
                print(f"  - {prereq}")

        print(f"\nConfidence: {result.confidence:.2f}")
        print("\n" + "="*60)


if __name__ == "__main__":
    main()
