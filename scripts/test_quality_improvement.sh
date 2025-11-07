#!/bin/bash

# Quality Improvement Testing Script
# This script helps you test improvements by running before/after comparisons

echo "======================================================================"
echo "🧪 Quality Improvement Testing Script"
echo "======================================================================"
echo ""

# Test queries (add your own failing queries here)
declare -a test_queries=(
    "How do I integrate MS Teams with Watermelon?"
    "How do I create a no-code block and test it?"
    "What are the security features in Watermelon?"
    "How do I set up automated responses?"
)

echo "📋 Test queries to run:"
for i in "${!test_queries[@]}"; do
    echo "  $((i+1)). ${test_queries[$i]}"
done
echo ""

read -p "Run diagnostic on all test queries? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo ""
    echo "======================================================================"
    echo "Running diagnostics..."
    echo "======================================================================"

    for query in "${test_queries[@]}"; do
        echo ""
        echo "----------------------------------------------------------------------"
        echo "Query: $query"
        echo "----------------------------------------------------------------------"
        python scripts/diagnose_quality.py "$query"
        echo ""
        read -p "Press Enter to continue to next query..."
    done

    echo ""
    echo "======================================================================"
    echo "✅ Diagnostic complete!"
    echo "======================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Review the diagnostic output above"
    echo "  2. Follow the suggestions in QUALITY_IMPROVEMENT_GUIDE.md"
    echo "  3. Make improvements (start with metadata enrichment)"
    echo "  4. Run this script again to verify improvements"
    echo ""
fi
