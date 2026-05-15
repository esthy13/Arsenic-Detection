# Arsenic-Detection
Data-driven (arsenic) contamination detection project for the course Machine Learning for Water Distribution System at the university of Bielefeld (a.a. 2025-2026)

# How to run the project

0. On your first time create a virtual environment
    ```bash
    python -m venv .venv
    ```
1. Activate the virtual environment .venv
    on linux:
    ```bash
    source .venv/bin/activate
    ```
    on windows:
    ```ps
    .venv\Scripts\activate
    ```
3. Install the needed libraries from the requirements file
    ```bash
    pip install -r requirements.txt
    ```

**Remember**
If you install new libraries on your local environment before committing and 
pushing to the repository do:
```bash
python -m pip freeze > requirements.txt
```