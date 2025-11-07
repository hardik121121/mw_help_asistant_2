"""
Configuration management for Watermelon Documentation Assistant.
Loads and validates all environment variables and provides typed access to settings.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
(CACHE_DIR / "images").mkdir(exist_ok=True)


class Settings(BaseSettings):
    """Main settings class that loads all configuration from environment variables."""

    # API Keys
    openai_api_key: str
    pinecone_api_key: str
    pinecone_environment: str = "us-east-1"
    cohere_api_key: str
    groq_api_key: str

    # Document Processing
    chunk_size: int = 1500
    chunk_overlap: int = 300
    min_chunk_size: int = 200

    # Heading detection
    heading_1_size: int = 20
    heading_2_size: int = 16
    heading_3_size: int = 14
    heading_4_size: int = 12

    # Retrieval Config
    vector_top_k: int = 50  # Increased from 30 for better coverage
    bm25_top_k: int = 50    # Increased from 30 for better coverage
    rerank_top_k: int = 20  # Increased from 10 for better final results
    rerank_model: str = "rerank-english-v3.0"
    rrf_k: int = 60
    page_proximity: int = 3

    # Generation Config
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 8192
    llm_context_window: int = 128000
    enable_query_decomposition: bool = True
    max_sub_questions: int = 4
    query_complexity_threshold: int = 10

    # Database Config
    pinecone_index_name: str = "watermelon-docs-v2"
    pinecone_dimension: int = 3072
    pinecone_metric: str = "cosine"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Image Config
    max_images_per_page: int = 5
    max_total_images: int = 15

    # Paths
    pdf_path: Path = DATA_DIR / "helpdocs.pdf"
    cache_dir: Path = CACHE_DIR
    image_cache_dir: Path = CACHE_DIR / "images"
    processed_pdf_path: Path = CACHE_DIR / "docling_processed.json"
    chunks_path: Path = CACHE_DIR / "hierarchical_chunks.json"
    embeddings_path: Path = CACHE_DIR / "hierarchical_embeddings.pkl"

    # Logging
    log_level: str = "INFO"
    enable_debug_mode: bool = False
    enable_query_logging: bool = True
    log_file: Path = LOGS_DIR / "app.log"

    # Performance
    embedding_batch_size: int = 100
    pinecone_batch_size: int = 100
    embedding_rate_limit: float = 0.1
    cohere_rate_limit: float = 0.05
    groq_rate_limit: float = 0.01
    enable_query_cache: bool = True
    cache_expiry: int = 3600

    # Evaluation
    test_queries_path: Path = Path("tests/test_queries.json")
    evaluation_output_dir: Path = Path("tests/results")
    track_retrieval_metrics: bool = True
    track_generation_metrics: bool = True
    track_latency: bool = True
    track_cost: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @field_validator("openai_api_key", "pinecone_api_key", "cohere_api_key", "groq_api_key")
    @classmethod
    def check_not_placeholder(cls, v: str) -> str:
        """Ensure API keys are not placeholder values."""
        if "xxx" in v.lower() or v.startswith("sk-proj-xxx"):
            raise ValueError(
                f"API key contains placeholder value. "
                f"Please set real API key in .env file."
            )
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        # We can't access chunk_size from values in Pydantic V2, so just do basic check
        if v >= 2000:  # Max reasonable overlap
            raise ValueError(f"chunk_overlap ({v}) is too large")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}, got: {v}")
        return v_upper

    @field_validator("pinecone_metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        """Ensure metric is valid."""
        valid_metrics = {"cosine", "euclidean", "dotproduct"}
        if v not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}, got: {v}")
        return v

    def validate_config(self) -> dict:
        """Validate all settings and return status report."""
        report = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        # Check API keys are set
        api_keys = {
            "OpenAI": self.openai_api_key,
            "Pinecone": self.pinecone_api_key,
            "Cohere": self.cohere_api_key,
            "Groq": self.groq_api_key,
        }

        for name, key in api_keys.items():
            if not key or len(key) < 10:
                report["errors"].append(f"{name} API key not set or invalid")
                report["valid"] = False

        # Check PDF exists
        if not self.pdf_path.exists():
            report["warnings"].append(f"PDF not found at: {self.pdf_path}")

        # Check chunk size configuration
        if self.chunk_overlap >= self.chunk_size:
            report["errors"].append(
                f"Chunk overlap ({self.chunk_overlap}) must be less than "
                f"chunk size ({self.chunk_size})"
            )
            report["valid"] = False

        return report

    def print_summary(self):
        """Print a summary of current configuration."""
        print("\n" + "="*60)
        print("📋 CONFIGURATION SUMMARY")
        print("="*60)

        print("\n🔑 API Configuration:")
        print(f"   OpenAI: {'✓ Set' if self.openai_api_key else '✗ Missing'}")
        print(f"   Pinecone: {'✓ Set' if self.pinecone_api_key else '✗ Missing'}")
        print(f"   Cohere: {'✓ Set' if self.cohere_api_key else '✗ Missing'}")
        print(f"   Groq: {'✓ Set' if self.groq_api_key else '✗ Missing'}")

        print("\n📄 Document Processing:")
        print(f"   Chunk Size: {self.chunk_size} chars")
        print(f"   Chunk Overlap: {self.chunk_overlap} chars")
        print(f"   PDF Path: {self.pdf_path}")
        print(f"   PDF Exists: {'✓ Yes' if self.pdf_path.exists() else '✗ No'}")

        print("\n🔍 Retrieval Configuration:")
        print(f"   Vector Top-K: {self.vector_top_k}")
        print(f"   BM25 Top-K: {self.bm25_top_k}")
        print(f"   Rerank Top-K: {self.rerank_top_k}")
        print(f"   Page Proximity: ±{self.page_proximity} pages")

        print("\n🤖 Generation Configuration:")
        print(f"   Model: {self.llm_model}")
        print(f"   Temperature: {self.llm_temperature}")
        print(f"   Max Tokens: {self.llm_max_tokens}")
        print(f"   Query Decomposition: {'✓ Enabled' if self.enable_query_decomposition else '✗ Disabled'}")

        print("\n💾 Database Configuration:")
        print(f"   Index Name: {self.pinecone_index_name}")
        print(f"   Dimension: {self.pinecone_dimension}")
        print(f"   Metric: {self.pinecone_metric}")

        print("\n📊 Performance:")
        print(f"   Embedding Batch: {self.embedding_batch_size}")
        print(f"   Query Cache: {'✓ Enabled' if self.enable_query_cache else '✗ Disabled'}")

        print("\n" + "="*60 + "\n")


# Global settings instance
try:
    settings = Settings()
except Exception as e:
    print(f"\n⚠️  Could not load settings: {e}")
    print("Please ensure .env file exists with valid API keys.\n")
    settings = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    if settings is None:
        raise RuntimeError(
            "Settings not initialized. Please ensure .env file exists with valid API keys."
        )
    return settings


if __name__ == "__main__":
    """Run configuration validation when executed directly."""
    try:
        config = Settings()
        config.print_summary()

        print("\n🔍 Validating configuration...")
        report = config.validate_config()

        if report["valid"]:
            print("✅ Configuration is valid!")
        else:
            print("❌ Configuration has errors:")
            for error in report["errors"]:
                print(f"   - {error}")

        if report["warnings"]:
            print("\n⚠️  Warnings:")
            for warning in report["warnings"]:
                print(f"   - {warning}")

    except Exception as e:
        print(f"\n❌ Failed to load configuration: {e}")
        exit(1)
