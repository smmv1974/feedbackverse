"""Setup configuration for Feedbackverse Multi-Platform Review Analysis."""

from setuptools import setup, find_packages

setup(
    name="feedbackverse",
    version="1.0.0",
    description="Multi-Platform Review Aggregation and LLM-Powered Analysis",
    author="Research Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "scikit-learn>=1.3.0",
        "sentence-transformers>=2.2.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "google-generativeai>=0.3.0",
        "groq>=0.4.0",
        "openai>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "feedbackverse-pipeline=scripts.run_pipeline_with_audit:main",
        ],
    },
)
