# ML_OPS_Demo2

# Telecom Customer Churn -- ML + LLMOps

This project demonstrates a machine learning and LLMOps workflow for
telecom customer churn analysis. It uses customer account information to
predict churn, tracks machine learning experiments with MLflow, and uses
an LLM to generate customer-specific retention recommendations.

## 1. Project Structure

``` text
ML_OPS_Demo2/
│
├── artifacts/                     # Saved model and other generated artifacts
├── data/
│   └── telco_customer_churn.csv   # Telecom customer churn dataset
├── mlruns/                        # MLflow experiment artifacts (generated)
├── myenv/                         # Python virtual environment (local only)
├── prompts/
│   ├── prompt_v1.txt              # Initial retention recommendation prompt
│   └── prompt_v2.txt              # Improved retention recommendation prompt
├── src/
│   ├── data_processing.py         # Data cleaning and preprocessing
│   ├── datacollection.py          # Data collection utilities
│   ├── llm_ops_mlflow.py           # LLMOps tracking and evaluation
│   ├── ml_experiments.py          # ML model experiments and MLflow logging
│   ├── ml_model.py                # Model training and model-related functions
│   └── ml_prediction_service.py   # Prediction service
├── .gitignore
├── LICENSE
├── mlflow.db                      # Local MLflow tracking database (not committed)
└── README.md
```

## 2. Create a GitHub Repository

1.  Sign in to [GitHub](https://github.com/).
2.  Select **New repository**.
3.  Enter a repository name, for example
    `telecom-customer-churn-ml-llmops`.
4.  Add a short description, such as
    `Telecom customer churn prediction and LLM-based retention recommendations`.
5.  Choose **Public** or **Private**, depending on your sharing
    requirements.
6.  For an existing local project, leave **Add a README file**, **Add
    .gitignore**, and **Choose a license** unchecked if those files
    already exist locally.
7.  Select **Create repository**.
8.  Copy the repository URL. It will look similar to:

``` text
https://github.com/<your-username>/telecom-customer-churn-ml-llmops.git
```

## 3. Open the Project in VS Code

Open the `ML_OPS_Demo2` project folder in VS Code.

Open the integrated terminal:

**Terminal → New Terminal**

Confirm that the terminal is in the project root folder. For example:

``` powershell
cd "C:\path\to\ML_OPS_Demo2"
```

Check the files and folders:

``` powershell
dir
```

## 4. Configure Git

Check whether Git is installed:

``` powershell
git --version
```

If Git is installed, configure your name and email. Use the name and
email associated with your Git commits:

``` powershell
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

You generally need to do this only once on a computer.

## 5. Initialize Git in the Project

From the project root, initialize a local Git repository:

``` powershell
git init
```

Check the repository status:

``` powershell
git status
```

Git now tracks the project as a local repository, but the files have not
yet been uploaded to GitHub.

## 6. Configure `.gitignore`

Before staging files, make sure local environments, generated tracking
data and private files are excluded.

Add or update the `.gitignore` file in the project root:

``` gitignore
# Python
__pycache__/
*.py[cod]
*.pyo

# Virtual environments
myenv/
venv/
.venv/

# Environment variables and secrets
.env
.env.*
!.env.example

# MLflow local tracking data
mlruns/
mlflow.db
mlflow.db-shm
mlflow.db-wal/

# Jupyter
.ipynb_checkpoints/

# Editor and operating system files
.vscode/
.DS_Store
Thumbs.db
```

Review `artifacts/` before committing. If it contains large generated
models or datasets, consider storing them separately rather than
committing them to Git. Do not ignore the entire folder if it contains
small, essential project files that the repository needs.

**Data privacy check:** Before uploading `telco_customer_churn.csv`,
confirm that it is permitted to share. Do not push real customer
records, personally identifiable information, credentials, API keys or
confidential company data to a public repository. If the dataset cannot
be shared, exclude it and document how to obtain an approved copy.

## 7. Stage the Project Files

Stage the files that should be included in the repository:

``` powershell
git add .
```

Review exactly what is staged:

``` powershell
git status
```

If you notice files that should not be uploaded, add them to
`.gitignore`, then unstage them if necessary:

``` powershell
git rm --cached -r myenv mlruns
git rm --cached mlflow.db
```

Run these removal commands only if those paths were previously staged or
tracked. They remove files from Git's index, not from your local working
folder.

## 8. Commit the Files

Create the first commit:

``` powershell
git commit -m "Initial commit: Telecom churn ML and LLMOps project"
```

A commit saves a snapshot of the staged project files in your local Git
repository. It does not upload them to GitHub yet.

## 9. Connect the Local Repository to GitHub

Set the default branch name to `main`:

``` powershell
git branch -M main
```

Add the GitHub repository as the remote named `origin`. Replace the
example URL with the URL copied from your GitHub repository:

``` powershell
git remote add origin https://github.com/<your-username>/telecom-customer-churn-ml-llmops.git
```

Verify the remote:

``` powershell
git remote -v
```

If `origin` is already configured, do not add it again. To correct its
URL, use:

``` powershell
git remote set-url origin https://github.com/<your-username>/telecom-customer-churn-ml-llmops.git
```

## 10. Push the Files to GitHub

Upload the local `main` branch to GitHub:

``` powershell
git push -u origin main
```

If prompted, complete GitHub authentication using the sign-in method
offered by Git Credential Manager or your configured Git authentication
method.

The `-u` option sets `origin/main` as the upstream branch. For
subsequent pushes, you can usually run:

``` powershell
git push
```

## 11. Verify the Repository

1.  Open your GitHub repository in the browser.
2.  Refresh the page.
3.  Confirm that the expected source files, `prompts/`, `.gitignore`,
    `LICENSE` and `README.md` are visible.
4.  Confirm that local-only folders and files such as `myenv/`,
    `mlruns/` and `mlflow.db` have not been uploaded.
5.  Check that the dataset and artifacts are included only if they are
    safe and appropriate to share.

## 12. Commands for Future Updates

Whenever you modify the project and want to upload the changes:

``` powershell
git status
git add .
git commit -m "Describe the changes made"
git push
```

### Git workflow summary

``` text
Create GitHub repository
          |
          v
Open local project in VS Code
          |
          v
git init
          |
          v
Configure .gitignore
          |
          v
git add .
          |
          v
git commit -m "Initial commit"
          |
          v
git branch -M main
          |
          v
git remote add origin <GitHub repository URL>
          |
          v
git push -u origin main
          |
          v
Verify files on GitHub
```

**Important:** `git add` stages files, `git commit` saves them locally,
and `git push` uploads committed changes to the remote GitHub
repository.

