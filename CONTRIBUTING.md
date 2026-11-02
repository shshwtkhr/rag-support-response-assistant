# Contributing to Support Response Assistant

Thank you for your interest in contributing to **Support Response Assistant**! We welcome contributions from the community.

---

## Code of Conduct

This project adheres to the Contributor Covenant [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

---

## Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork locally**:
   ```bash
   git clone https://github.com/<your-username>/rag-support-response-assistant.git
   cd rag-support-response-assistant
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest ruff
   ```

---

## Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes with clear, descriptive commit messages.
3. Ensure all tests pass:
   ```bash
   pytest test_rag.py -v
   ```
4. Run the code linter:
   ```bash
   ruff check app.py src test_rag.py
   ```
5. Push to your branch and open a Pull Request.

---

## Coding Guidelines

- Write clean, readable, documented code following PEP 8.
- Use explicit exception chaining (`raise ... from err`) when raising exceptions inside `except` blocks.
- Add unit tests for any new features or bug fixes in `test_*.py`.
- Keep the system modular, maintaining clear separation between document loading, indexing, retrieval, and LLM generation.

---

## Submitting Pull Requests

- Provide a clear PR description explaining the motivation and changes.
- Verify that GitHub Actions CI checks pass on your PR.
- We will review your PR and provide feedback as soon as possible!
