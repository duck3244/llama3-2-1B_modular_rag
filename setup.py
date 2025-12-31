"""
Setup script for Modular RAG System
Ensures Python 3.10 compatibility
"""
import sys
from setuptools import setup, find_packages

# Python version check
if sys.version_info < (3, 10) or sys.version_info >= (3, 11):
    sys.exit("Error: This project requires Python 3.10.x")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llama-modular-rag",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A CPU-optimized RAG system with Llama 3.2 and LangGraph for Python 3.10",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llama-modular-rag",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10,<3.11",
    install_requires=requirements,
    keywords="rag llama langchain langgraph nlp korean ai machine-learning",
)
