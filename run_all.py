#!/usr/bin/env python3
"""
Portfolio Launcher — Insurance AI Engineering Portfolio
Guides you through running all 4 projects interactively.

Usage:
    python run_all.py            # Interactive menu
    python run_all.py --demos    # Open all 4 frontend demos
    python run_all.py --test     # Run all test suites
    python run_all.py --check    # Check Python version and dependencies
"""

import os
import sys
import subprocess
import platform
import webbrowser
import time
import argparse
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

PROJECTS = [
    {
        "id": 1,
        "key": "claims",
        "name": "Claims Triage & Fraud Signal Agent",
        "dir": "claims-triage-agent/backend",
        "frontend": "claims-triage-agent/frontend/index.html",
        "port": 8001,
        "entry": "main:app",
        "color": "\033[91m",   # red
        "icon": "🔍",
        "description": "Agentic fraud detection pipeline · 7 stages · PII redaction · HITL escalation",
        "api_example": 'curl -s -X POST http://localhost:8001/api/v1/claims/process -H "Content-Type: application/json" -H "X-User-Role: adjuster" -d \'{"claim_text":"My car was rear-ended on the highway. No injuries, damage to front bumper.","policy_number":"LM-4419823","claimant_name":"James Smith","incident_date":"2024-05-01","claim_type":"auto"}\'',
    },
    {
        "id": 2,
        "key": "uw",
        "name": "Underwriting Copilot (RAG)",
        "dir": "uw-copilot/backend",
        "frontend": "uw-copilot/frontend/index.html",
        "port": 8002,
        "entry": "main:app",
        "color": "\033[94m",   # blue
        "icon": "📚",
        "description": "RAG knowledge retrieval · Azure AI Search · Cohere rerank · Citation UI",
        "api_example": 'curl -s -X POST http://localhost:8002/api/v1/copilot/query -H "Content-Type: application/json" -d \'{"query_text":"What is the maximum TIV for a single habitational risk without reinsurance?"}\'',
    },
    {
        "id": 3,
        "key": "gateway",
        "name": "Governed AI Gateway",
        "dir": "ai-gateway/backend",
        "frontend": "ai-gateway/frontend/index.html",
        "port": 8003,
        "entry": "main:app",
        "color": "\033[92m",   # green
        "icon": "🛡️",
        "description": "JWT auth · RBAC · PII redaction · Audit log · Cost tracking · Policy engine",
        "api_example": 'curl -s http://localhost:8003/health',
    },
    {
        "id": 4,
        "key": "eval",
        "name": "LLM Eval & Regression Framework",
        "dir": "llm-eval-framework/backend",
        "frontend": "llm-eval-framework/frontend/index.html",
        "port": 8004,
        "entry": "api.routes:app",
        "color": "\033[93m",   # yellow
        "icon": "📊",
        "description": "8 metrics · CI regression gate · Model comparison · 22 insurance eval cases",
        "api_example": 'curl -s http://localhost:8004/api/v1/runs | python3 -m json.tool',
    },
]

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
CYAN  = "\033[96m"
WHITE = "\033[97m"

ROOT = Path(__file__).parent


# ── Helpers ───────────────────────────────────────────────────────────────────

def clear():
    os.system("cls" if platform.system() == "Windows" else "clear")

def print_header():
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════╗
║     Insurance AI Engineering Portfolio — Run Launcher        ║
║     Senior Applied AI Engineer · Liberty Mutual Insurance    ║
╚══════════════════════════════════════════════════════════════╝{RESET}
""")

def print_project(p, indent=4):
    pad = " " * indent
    print(f"{pad}{p['color']}{BOLD}{p['icon']}  {p['id']}. {p['name']}{RESET}")
    print(f"{pad}   {DIM}{p['description']}{RESET}")
    print(f"{pad}   {DIM}Backend: localhost:{p['port']}  |  Dir: {p['dir']}{RESET}")

def run_cmd(cmd: list[str], cwd: Path | None = None, capture: bool = False):
    kwargs = {"cwd": cwd}
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    return subprocess.run(cmd, **kwargs)

def open_browser(path: str):
    abs_path = ROOT / path
    if abs_path.exists():
        url = abs_path.as_uri()
        print(f"  Opening {url}")
        webbrowser.open(url)
    else:
        print(f"  ⚠  File not found: {abs_path}")

def check_python():
    major, minor = sys.version_info.major, sys.version_info.minor
    ok = major == 3 and minor >= 11
    status = "✅" if ok else "❌"
    print(f"  {status} Python {major}.{minor} {'(OK — 3.11+ required)' if ok else '(NEEDS UPGRADE — requires 3.11+)'}")
    return ok

def check_deps(project_dir: str) -> bool:
    req = ROOT / project_dir / "requirements.txt"
    if not req.exists():
        print(f"  ⚠  No requirements.txt found in {project_dir}")
        return False

    # Check if core deps are importable
    result = run_cmd(
        [sys.executable, "-c", "import fastapi, pydantic, uvicorn"],
        capture=True,
    )
    return result.returncode == 0

def install_deps(project_dir: str):
    req = ROOT / project_dir / "requirements.txt"
    print(f"\n  Installing dependencies for {project_dir}...")
    print(f"  {DIM}pip install -r {req}{RESET}\n")
    run_cmd([sys.executable, "-m", "pip", "install", "-r", str(req)])

def start_backend(project: dict) -> subprocess.Popen:
    """Start a backend server in the background."""
    cwd = ROOT / project["dir"]
    cmd = [
        sys.executable, "-m", "uvicorn",
        project["entry"],
        "--host", "0.0.0.0",
        "--port", str(project["port"]),
        "--reload",
    ]
    print(f"  Starting {project['name']} backend on port {project['port']}...")
    proc = subprocess.Popen(cmd, cwd=cwd)
    time.sleep(2)  # Give server time to start
    return proc


# ── Actions ───────────────────────────────────────────────────────────────────

def action_open_demos():
    """Open all 4 frontend demos in the browser."""
    print(f"\n{BOLD}Opening all 4 frontend demos...{RESET}\n")
    print("  These run entirely in the browser — no server needed.\n")
    for p in PROJECTS:
        print_project(p)
        open_browser(p["frontend"])
        time.sleep(0.5)
    print(f"\n{CYAN}✅ All 4 demos opened!{RESET}")
    print(f"{DIM}Each demo is fully interactive with pre-loaded data.{RESET}\n")

def action_open_single_demo(project_id: int):
    """Open one project's demo."""
    p = next((x for x in PROJECTS if x["id"] == project_id), None)
    if not p:
        print(f"  ❌ Project {project_id} not found")
        return
    print(f"\n  Opening {p['name']} demo...")
    open_browser(p["frontend"])

def action_run_tests():
    """Run all test suites."""
    print(f"\n{BOLD}Running all test suites...{RESET}\n")

    results = {}
    for p in PROJECTS:
        test_dir = ROOT / p["dir"].replace("/backend", "") / "backend" / "tests"
        if not test_dir.exists():
            print(f"  ⚠  No tests directory found for {p['name']}")
            continue

        print(f"\n{p['color']}{BOLD}{'─'*60}{RESET}")
        print(f"{p['color']}{BOLD}  {p['icon']} {p['name']}{RESET}")
        print(f"{p['color']}{'─'*60}{RESET}\n")

        result = run_cmd(
            [sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=short", "-q"],
            cwd=ROOT / p["dir"],
        )
        results[p["name"]] = result.returncode == 0

    print(f"\n{BOLD}{'═'*60}")
    print("  Test Results Summary")
    print(f"{'═'*60}{RESET}\n")
    for name, passed in results.items():
        status = f"{'✅ PASSED' if passed else '❌ FAILED'}"
        color  = "\033[92m" if passed else "\033[91m"
        print(f"  {color}{status}{RESET}  {name}")

    all_passed = all(results.values())
    print(f"\n  {'✅ All tests passed!' if all_passed else '❌ Some tests failed — see output above'}\n")

def action_check_environment():
    """Check Python version and key dependencies."""
    print(f"\n{BOLD}Environment Check{RESET}\n")
    py_ok = check_python()

    print(f"\n  {BOLD}Key packages:{RESET}")
    packages = ["fastapi", "uvicorn", "pydantic", "pytest", "httpx", "openai"]
    for pkg in packages:
        result = run_cmd([sys.executable, "-c", f"import {pkg}; print({pkg}.__version__)"], capture=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"  ✅ {pkg} {version}")
        else:
            print(f"  ❌ {pkg} — not installed")

    print(f"\n  {BOLD}Project structure:{RESET}")
    for p in PROJECTS:
        backend_dir = ROOT / p["dir"]
        frontend    = ROOT / p["frontend"]
        b_ok = "✅" if backend_dir.exists() else "❌"
        f_ok = "✅" if frontend.exists() else "❌"
        print(f"  {b_ok} {p['name']} backend  |  {f_ok} frontend")

def action_install_all():
    """Install dependencies for all projects."""
    print(f"\n{BOLD}Installing dependencies for all projects...{RESET}")
    print(f"{DIM}This may take several minutes.{RESET}\n")

    for p in PROJECTS:
        req = ROOT / p["dir"] / "requirements.txt"
        if not req.exists():
            print(f"  ⚠  Skipping {p['name']} — no requirements.txt")
            continue
        print(f"\n{p['color']}{BOLD}  {p['icon']} {p['name']}{RESET}")
        run_cmd([sys.executable, "-m", "pip", "install", "-r", str(req)])

    print(f"\n{CYAN}✅ All dependencies installed!{RESET}\n")

def action_start_backend(project_id: int):
    """Start a single backend server (blocking — runs in foreground)."""
    p = next((x for x in PROJECTS if x["id"] == project_id), None)
    if not p:
        print(f"  ❌ Project {project_id} not found")
        return

    cwd = ROOT / p["dir"]
    if not cwd.exists():
        print(f"  ❌ Backend directory not found: {cwd}")
        print(f"  Make sure you're running from the repo root.")
        return

    print(f"\n{p['color']}{BOLD}  {p['icon']} Starting {p['name']}{RESET}")
    print(f"  {DIM}Directory: {cwd}{RESET}")
    print(f"  {DIM}Port:      {p['port']}{RESET}")
    print(f"  {DIM}API docs:  http://localhost:{p['port']}/docs{RESET}")
    print(f"\n  {DIM}Press Ctrl+C to stop{RESET}\n")

    cmd = [
        sys.executable, "-m", "uvicorn",
        p["entry"],
        "--host", "0.0.0.0",
        "--port", str(p["port"]),
        "--reload",
    ]
    try:
        subprocess.run(cmd, cwd=cwd)
    except KeyboardInterrupt:
        print(f"\n\n  {p['name']} stopped.")

def action_show_api_example(project_id: int):
    """Show an API call example for a project."""
    p = next((x for x in PROJECTS if x["id"] == project_id), None)
    if not p:
        return
    print(f"\n  {BOLD}Example API call for {p['name']}:{RESET}")
    print(f"  {DIM}(requires backend running on port {p['port']}){RESET}\n")
    print(f"  {CYAN}{p['api_example']}{RESET}\n")


# ── Interactive Menu ──────────────────────────────────────────────────────────

def interactive_menu():
    while True:
        clear()
        print_header()

        print(f"  {BOLD}Projects:{RESET}\n")
        for p in PROJECTS:
            print_project(p)
            print()

        print(f"  {BOLD}Actions:{RESET}\n")
        print(f"  {CYAN}[D]{RESET}  Open ALL demos in browser (no server needed)")
        print(f"  {CYAN}[1-4]{RESET} Open specific project demo")
        print(f"  {CYAN}[S1-S4]{RESET} Start backend server (S1=claims, S2=uw, S3=gateway, S4=eval)")
        print(f"  {CYAN}[T]{RESET}  Run all test suites")
        print(f"  {CYAN}[I]{RESET}  Install all dependencies")
        print(f"  {CYAN}[C]{RESET}  Check environment")
        print(f"  {CYAN}[Q]{RESET}  Quit\n")

        choice = input(f"  {BOLD}Enter choice:{RESET} ").strip().upper()

        if choice == "Q":
            print(f"\n  {CYAN}Good luck with the interview! 🚀{RESET}\n")
            break
        elif choice == "D":
            action_open_demos()
            input(f"\n  {DIM}Press Enter to return to menu...{RESET}")
        elif choice in ("1", "2", "3", "4"):
            action_open_single_demo(int(choice))
            time.sleep(1)
        elif choice in ("S1", "S2", "S3", "S4"):
            project_id = int(choice[1])
            action_start_backend(project_id)
            input(f"\n  {DIM}Press Enter to return to menu...{RESET}")
        elif choice == "T":
            action_run_tests()
            input(f"\n  {DIM}Press Enter to return to menu...{RESET}")
        elif choice == "I":
            action_install_all()
            input(f"\n  {DIM}Press Enter to return to menu...{RESET}")
        elif choice == "C":
            action_check_environment()
            input(f"\n  {DIM}Press Enter to return to menu...{RESET}")
        elif choice.startswith("A") and choice[1:].isdigit():
            action_show_api_example(int(choice[1:]))
            input(f"\n  {DIM}Press Enter to return to menu...{RESET}")
        else:
            print(f"\n  ❌ Unknown choice: {choice}")
            time.sleep(1)


# ── CLI Mode ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Insurance AI Portfolio Launcher")
    parser.add_argument("--demos",  action="store_true", help="Open all 4 frontend demos")
    parser.add_argument("--test",   action="store_true", help="Run all test suites")
    parser.add_argument("--check",  action="store_true", help="Check environment")
    parser.add_argument("--install",action="store_true", help="Install all dependencies")
    parser.add_argument("--start",  type=int, metavar="N", help="Start backend N (1-4)")
    args = parser.parse_args()

    if args.demos:
        action_open_demos()
    elif args.test:
        action_run_tests()
    elif args.check:
        action_check_environment()
    elif args.install:
        action_install_all()
    elif args.start:
        action_start_backend(args.start)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
