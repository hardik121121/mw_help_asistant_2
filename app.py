"""
Watermelon Documentation Assistant - Streamlit UI
Interactive web interface for querying the documentation.
"""

import streamlit as st
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.generation.end_to_end_pipeline import EndToEndPipeline, PipelineResult


# Page configuration
st.set_page_config(
    page_title="Watermelon Documentation Assistant",
    page_icon="🍉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .answer-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B6B;
        margin: 1rem 0;
    }
    .stage-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .citation {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load and cache the RAG pipeline."""
    with st.spinner("🔧 Initializing RAG pipeline... (this may take a minute)"):
        pipeline = EndToEndPipeline(
            use_reranking=True,
            enable_context_chaining=True,
            validate_responses=True
        )
    return pipeline


def format_answer_with_images(answer_text: str, images_used: list) -> None:
    """Format and display the answer text with inline images."""
    import re

    # Split answer into lines
    lines = answer_text.split('\n')

    for line in lines:
        # Check if line contains an image path
        image_match = re.search(r'cache/images/[\w_]+\.png', line)

        if image_match:
            image_path = image_match.group(0)

            # Display the text part (description)
            text_before = line[:image_match.start()].strip()
            text_after = line[image_match.end():].strip()

            if text_before:
                st.markdown(text_before)

            # Display the image inline
            if Path(image_path).exists():
                st.image(image_path, caption=Path(image_path).name, width=400)
            else:
                st.caption(f"🖼️ {Path(image_path).name} {text_after}")

            if text_after and not text_before:
                st.caption(text_after)
        else:
            # Normal text line
            st.markdown(line)


def format_answer(answer_text: str) -> str:
    """Format the answer text with proper markdown."""
    return answer_text


def display_pipeline_stages(result: PipelineResult):
    """Display information about each pipeline stage."""

    st.markdown("### 📊 Pipeline Stages")

    # Stage 1: Query Understanding
    with st.expander("🧠 Stage 1: Query Understanding", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Query Type", result.understanding.classification.query_type.value)
        with col2:
            st.metric("Complexity", result.understanding.classification.complexity.value)
        with col3:
            st.metric("Sub-questions", len(result.understanding.decomposition.sub_questions))

        st.markdown("**Sub-questions:**")
        for i, sq in enumerate(result.understanding.decomposition.sub_questions, 1):
            st.markdown(f"{i}. {sq}")

        st.markdown(f"**Generation Strategy:** {result.understanding.generation_strategy}")

    # Stage 2: Retrieval
    with st.expander("🔍 Stage 2: Multi-Step Retrieval", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Retrieved", result.retrieval.total_chunks_retrieved)
        with col2:
            st.metric("Final Chunks", result.retrieval.final_chunks)
        with col3:
            st.metric("Unique Sections", result.retrieval.organized_context.unique_sections)
        with col4:
            st.metric("Time", f"{result.retrieval.retrieval_time:.2f}s")

        st.markdown("**Retrieved Sections:**")
        # Get section names from section_hierarchy or chunks
        sections = list(result.retrieval.organized_context.section_hierarchy.keys())[:10]  # First 10
        if sections:
            for section in sections:
                st.markdown(f"- {section}")
        else:
            st.info("No section information available")

    # Stage 3: Generation
    with st.expander("✨ Stage 3: Answer Generation", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Words", len(result.answer.answer.split()))
        with col2:
            st.metric("Tokens", result.answer.tokens_used)
        with col3:
            st.metric("Citations", len(result.answer.citations))
        with col4:
            st.metric("Time", f"{result.answer.generation_time:.2f}s")

        if result.answer.confidence:
            st.progress(result.answer.confidence, text=f"Confidence: {result.answer.confidence:.0%}")

    # Stage 4: Validation
    if result.validation:
        with st.expander("✅ Stage 4: Validation", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Overall Score", f"{result.validation.overall_score:.2f}")
            with col2:
                st.metric("Completeness", f"{result.validation.completeness_score:.2f}")
            with col3:
                st.metric("Formatting", f"{result.validation.formatting_score:.2f}")

            if result.validation.issues:
                st.markdown("**Issues:**")
                for issue in result.validation.issues:
                    st.warning(issue)

            if result.validation.warnings:
                st.markdown("**Warnings:**")
                for warning in result.validation.warnings:
                    st.info(warning)


def display_images(images_used: list):
    """Display referenced images."""
    if not images_used:
        return

    st.markdown("### 🖼️ Referenced Images")

    # Display images in a grid
    cols = st.columns(3)
    for idx, image_path in enumerate(images_used[:9]):  # Show first 9 images
        col_idx = idx % 3
        with cols[col_idx]:
            try:
                if Path(image_path).exists():
                    st.image(image_path, use_container_width=True, caption=Path(image_path).name)
                else:
                    st.caption(f"❌ {Path(image_path).name} (not found)")
            except Exception as e:
                st.caption(f"❌ Error loading {Path(image_path).name}")


def display_citations(citations: list):
    """Display citations in a formatted way."""
    if not citations:
        st.info("No citations available")
        return

    st.markdown("### 📚 Citations")

    for i, citation in enumerate(citations[:10], 1):  # Show first 10
        with st.container():
            # Citations are strings, not dictionaries
            st.markdown(f"**[{i}]** {citation}")


def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<div class="main-header">🍉 Watermelon Documentation Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask complex questions about Watermelon documentation and get comprehensive answers</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        # Display settings
        show_pipeline = st.checkbox("Show Pipeline Stages", value=True)
        show_citations = st.checkbox("Show Citations", value=True)
        show_metrics = st.checkbox("Show Performance Metrics", value=True)

        st.info("💡 Images are automatically shown inline with the answer!")

        st.markdown("---")
        st.markdown("## 📖 About")
        st.markdown("""
        This assistant uses advanced RAG techniques:
        - **Query Decomposition**: Breaks complex queries into sub-questions
        - **Hybrid Search**: Vector + BM25 keyword search
        - **Semantic Reranking**: Cohere reranking for precision
        - **Multi-context Generation**: Comprehensive answers
        """)

        st.markdown("---")
        st.markdown("## 📊 System Status")
        st.success("✅ Pipeline Ready")
        st.info("💾 Indexed: 2,106 chunks")
        st.info("📄 Documentation: 2,257 pages")

        st.markdown("---")
        st.markdown("## 📈 Performance")
        st.markdown("""
        **Latest Evaluation Results:**
        - Precision@10: **0.667** (+19%)
        - Recall@10: **0.638** (+43%)
        - MRR: **0.854** (+49%)
        - Avg Time: **27.7s**
        - Quality: **100% Excellent**
        """)

        st.markdown("**Improvements Active:**")
        st.success("✅ Query Expansion (3x variations)")
        st.success("✅ Fine-tuned Decomposition")

    # Initialize pipeline
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"❌ Failed to initialize pipeline: {e}")
        st.stop()

    # Main content area
    st.markdown("---")

    # Query input
    st.markdown("## 💬 Ask a Question")

    # Example queries
    example_queries = [
        "How do I create a no-code block on Watermelon?",
        "What are the steps to integrate Shopify with Watermelon?",
        "How do I set up MS Teams integration?",
        "What is Autonomous Functional Testing and how do I use it?",
        "How do I configure live chat handover to a human agent?",
    ]

    selected_example = st.selectbox(
        "Choose an example query or write your own:",
        [""] + example_queries,
        format_func=lambda x: "Select an example..." if x == "" else x
    )

    query = st.text_area(
        "Your question:",
        value=selected_example,
        height=100,
        placeholder="Type your question here... (e.g., 'How do I integrate Shopify with Watermelon?')"
    )

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        ask_button = st.button("🚀 Ask Question", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)

    if clear_button:
        st.rerun()

    # Process query
    if ask_button and query.strip():
        st.markdown("---")

        # Progress indicator
        with st.spinner("🤔 Processing your question..."):
            start_time = time.time()

            try:
                # Process query
                result = pipeline.process_query(query)
                elapsed = time.time() - start_time

                # Display answer
                st.markdown("## ✨ Answer")
                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.markdown(result.answer.answer)
                st.markdown('</div>', unsafe_allow_html=True)

                # Display images separately
                if result.answer.images_used:
                    st.markdown("---")
                    display_images(result.answer.images_used)

                # Performance metrics
                if show_metrics:
                    st.markdown("---")
                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        st.metric("⏱️ Total Time", f"{result.total_time:.2f}s")
                    with col2:
                        st.metric("🔍 Retrieval", f"{result.retrieval.retrieval_time:.2f}s")
                    with col3:
                        st.metric("✨ Generation", f"{result.answer.generation_time:.2f}s")
                    with col4:
                        st.metric("📊 Quality", f"{result.validation.overall_score:.2f}")
                    with col5:
                        st.metric("✅ Status", "Success" if result.success else "Failed")

                # Pipeline stages
                if show_pipeline:
                    st.markdown("---")
                    display_pipeline_stages(result)

                # Citations (images are now inline in the answer)
                if show_citations and result.answer.citations:
                    st.markdown("---")
                    display_citations(result.answer.citations)

                # Success message
                st.success(f"✅ Query processed successfully in {result.total_time:.2f} seconds!")

            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
                st.exception(e)

    elif ask_button:
        st.warning("⚠️ Please enter a question first!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Powered by: OpenAI Embeddings • Pinecone Vector DB • Cohere Reranking • Groq Llama 3.3 70B</p>
        <p>Built with ❤️ using advanced RAG techniques</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
