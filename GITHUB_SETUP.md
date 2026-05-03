# GitHub Setup Guide — Insurance AI Portfolio

Complete step-by-step instructions for publishing this portfolio to GitHub.

---

## Step 1 — Install Git (if needed)

**Mac:**
```bash
# Install Homebrew first if you don't have it:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Then install git:
brew install git
```

**Windows:**  
Download from https://git-scm.com/download/win and run the installer.

**Verify:**
```bash
git --version
# Should print: git version 2.x.x
```

---

## Step 2 — Create a GitHub Account (if needed)

1. Go to https://github.com
2. Click **Sign up** and create a free account
3. Verify your email address

---

## Step 3 — Configure Git with Your Identity

```bash
git config --global user.name "Your Full Name"
git config --global user.email "your.email@example.com"

# Verify:
git config --global --list
```

---

## Step 4 — Organize Your Project Files

You should have this folder structure on your computer (from the Claude downloads):

```
insurance-ai-portfolio/           ← Create this folder
├── README.md                     ← From portfolio-hub/
├── run_all.py                    ← From portfolio-hub/
├── .gitignore                    ← From portfolio-hub/
├── .github/
│   └── workflows/
│       └── ci.yml                ← From portfolio-hub/
├── claims-triage-agent/          ← Entire project folder
│   ├── backend/
│   └── frontend/
├── uw-copilot/                   ← Entire project folder
│   ├── backend/
│   └── frontend/
├── ai-gateway/                   ← Entire project folder
│   ├── backend/
│   └── frontend/
└── llm-eval-framework/           ← Entire project folder
    ├── backend/
    ├── ci/
    └── frontend/
```

**To set this up:**

```bash
# Create the root folder
mkdir insurance-ai-portfolio
cd insurance-ai-portfolio

# Move your downloaded projects into it
# (adjust paths to where you saved the Claude downloads)
cp -r ~/Downloads/claims-triage-agent  .
cp -r ~/Downloads/uw-copilot           .
cp -r ~/Downloads/ai-gateway           .
cp -r ~/Downloads/llm-eval-framework   .

# Copy the portfolio hub files (README, run_all.py, .gitignore, .github/)
cp ~/Downloads/portfolio-hub/README.md   .
cp ~/Downloads/portfolio-hub/run_all.py  .
cp ~/Downloads/portfolio-hub/.gitignore  .
mkdir -p .github/workflows
cp ~/Downloads/portfolio-hub/.github/workflows/ci.yml .github/workflows/
```

---

## Step 5 — Create a GitHub Repository

1. Go to https://github.com/new
2. Fill in the form:
   - **Repository name:** `insurance-ai-portfolio`
   - **Description:** `Senior Applied AI Engineer portfolio — Claims Triage, UW Copilot, AI Gateway, LLM Eval Framework`
   - **Visibility:** ✅ **Public** (so interviewers can see it)
   - **Initialize:** ❌ Do NOT check "Add a README" (you already have one)
3. Click **Create repository**

---

## Step 6 — Push Your Code to GitHub

```bash
# Navigate to your portfolio root folder
cd insurance-ai-portfolio

# Initialize git
git init

# Add all files
git add .

# Check what will be committed (optional sanity check)
git status

# Create the first commit
git commit -m "Initial portfolio commit — all 4 AI engineering projects

Projects:
- Claims Triage & Fraud Signal Agent (agentic pipeline, LLM+ML fraud scoring)
- Underwriting Copilot RAG (Azure AI Search, Cohere rerank, citation UI)
- Governed AI Gateway (RBAC, PII redaction, audit log, cost tracking)
- LLM Eval & Regression Framework (8 metrics, CI gate, regression engine)"

# Connect to your GitHub repo (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/insurance-ai-portfolio.git

# Set the main branch
git branch -M main

# Push!
git push -u origin main
```

**Enter your GitHub credentials when prompted.**

> **Note:** If prompted for a password, GitHub no longer accepts account passwords for git push.  
> Use a Personal Access Token instead — see Step 6b below.

---

## Step 6b — Create a Personal Access Token (if needed)

1. Go to https://github.com/settings/tokens
2. Click **Generate new token (classic)**
3. Name: `portfolio-push`
4. Expiration: 90 days
5. Scopes: check ✅ **repo**
6. Click **Generate token**
7. Copy the token (starts with `ghp_...`)
8. Use it as your password when git push asks for credentials

---

## Step 7 — Verify Your Repository

1. Go to `https://github.com/YOUR_USERNAME/insurance-ai-portfolio`
2. You should see all 4 project folders and the README rendered
3. Click **Actions** tab — you should see the CI workflow running

---

## Step 8 — Enable GitHub Pages (Optional — Host the Demos Online)

You can host all 4 frontend demos publicly using GitHub Pages:

1. Go to your repo → **Settings** → **Pages**
2. Source: **Deploy from a branch**
3. Branch: `main` / root: `/`
4. Click **Save**

After a few minutes, your demos will be live at:
```
https://YOUR_USERNAME.github.io/insurance-ai-portfolio/claims-triage-agent/frontend/
https://YOUR_USERNAME.github.io/insurance-ai-portfolio/uw-copilot/frontend/
https://YOUR_USERNAME.github.io/insurance-ai-portfolio/ai-gateway/frontend/
https://YOUR_USERNAME.github.io/insurance-ai-portfolio/llm-eval-framework/frontend/
```

Add these URLs to your resume and LinkedIn!

---

## Step 9 — Update README with Your Details

Edit `README.md` and replace the placeholders at the bottom:

```markdown
## Contact

Built by [Your Name] — [your.email@example.com] — [LinkedIn URL]
```

```bash
git add README.md
git commit -m "Add contact details"
git push
```

---

## Step 10 — Polish Your GitHub Profile

1. Go to https://github.com/YOUR_USERNAME
2. Click **Edit profile**
3. Add: Name, Bio (`Senior Applied AI Engineer — Insurance Domain`), Location
4. Pin the `insurance-ai-portfolio` repo to your profile

---

## Common Issues & Fixes

**"Permission denied" on git push:**
```bash
# Use HTTPS instead of SSH:
git remote set-url origin https://github.com/YOUR_USERNAME/insurance-ai-portfolio.git
```

**"Repository not found" error:**
```bash
# Check your remote URL:
git remote -v
# Should show: https://github.com/YOUR_USERNAME/insurance-ai-portfolio.git
```

**Files too large (GitHub 100MB limit):**
```bash
# Check for large files:
find . -size +50M -not -path "./.git/*"

# If you have model weights or data files, add them to .gitignore
echo "*.pkl" >> .gitignore
echo "*.h5"  >> .gitignore
git add .gitignore && git commit -m "Add large file exclusions"
```

**Tests failing in CI:**
```bash
# Run tests locally first to see the errors:
cd claims-triage-agent/backend
pip install fastapi uvicorn pydantic pytest pytest-asyncio httpx
pytest tests/ -v
```

**Want to update a project after pushing:**
```bash
# Make your changes, then:
git add .
git commit -m "Update claims pipeline — improved fraud scoring"
git push
```

---

## What the Interviewer Sees

When you share `https://github.com/YOUR_USERNAME/insurance-ai-portfolio`:

1. **Root README** — clear project overview, JD mapping table, quick start
2. **CI badges** — green checkmarks show all tests pass
3. **4 project folders** — each with its own detailed README
4. **Code quality** — Pydantic schemas, typed interfaces, docstrings, tests
5. **GitHub Actions** — professional CI/CD setup they'd use in production

**Share the link in:**
- Your resume (Projects section)
- LinkedIn profile (Featured section)  
- Cover letter ("Please see my portfolio at...")
- The interview itself ("Let me walk you through the code")

---

## Quick Reference — Key URLs After Setup

| Resource | URL |
|----------|-----|
| GitHub repo | `https://github.com/YOUR_USERNAME/insurance-ai-portfolio` |
| Claims demo | `https://YOUR_USERNAME.github.io/insurance-ai-portfolio/claims-triage-agent/frontend/` |
| UW Copilot demo | `https://YOUR_USERNAME.github.io/insurance-ai-portfolio/uw-copilot/frontend/` |
| AI Gateway demo | `https://YOUR_USERNAME.github.io/insurance-ai-portfolio/ai-gateway/frontend/` |
| Eval Framework demo | `https://YOUR_USERNAME.github.io/insurance-ai-portfolio/llm-eval-framework/frontend/` |
