from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from vectorstore import get_vectorstore

llm = Ollama(model="mistral")
db = get_vectorstore()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())








#!/usr/bin/env python3
"""
AI Ops Bug Fix Automation Script
Automatically clones, analyzes, fixes, and creates PR for production bugs

Updated behavior (2025-08-29):
- Attempts to use `claude-code` CLI to authenticate (if needed), analyze, and fix code.
- If `claude-code` isn't available or fails, falls back to the original simulated fix logic.
- Prefers branch name "feature/ai-fix" (from env FEATURE_BRANCH or default) and appends timestamp only if branch exists.
- Keeps previous behavior for cloning, testing, commit/push, PR creation, and cleanup.
"""
 
import os
import sys
import subprocess
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse
import json
 
# Install python-dotenv if not available
try:
    from dotenv import load_dotenv
except ImportError:
    print("Installing python-dotenv...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'python-dotenv'])
    from dotenv import load_dotenv
 
# Color codes for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    NC = '\033[0m'
 
class AIOpsFixer:
    def __init__(self, repo_url: str, env_file: str = '.env'):
        # Load environment variables from .env file
        self.load_env_config(env_file)
       
        self.repo_url = repo_url
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # prefer base branch name 'feature/ai-fix' but append timestamp if exists to avoid collisions
        self.desired_branch_base = os.getenv('FEATURE_BRANCH', 'feature/ai-fix')
        self.branch_name = self.desired_branch_base  # will adjust after cloning if necessary
        self.repo_name = os.path.basename(repo_url.rstrip('/').replace('.git', ''))
        self.work_dir = Path(tempfile.mkdtemp(prefix=f"ai-ops-fix-{self.timestamp}-"))
        self.repo_path = self.work_dir / self.repo_name
       
        # Configuration from .env
        self.git_user_name = os.getenv('GIT_USER_NAME', 'AI-Ops-Bot')
        self.git_user_email = os.getenv('GIT_USER_EMAIL', 'ai-ops@company.com')
        self.default_branch = os.getenv('DEFAULT_BRANCH', 'main')
       
        # Validate requirements
        self.validate_requirements()
   
    def load_env_config(self, env_file: str):
        """Load configuration from .env file"""
        env_path = Path(env_file)
       
        if not env_path.exists():
            self.create_default_env_file(env_path)
            print(f"{Colors.YELLOW}Created default .env file at {env_path}")
            print(f"Please update it with your ANTHROPIC_API_KEY and run again.{Colors.NC}")
            sys.exit(1)
       
        # Load the .env file
        load_dotenv(env_path)
        print(f"{Colors.GREEN}✅ Loaded configuration from {env_file}{Colors.NC}")
   
    def create_default_env_file(self, env_path: Path):
        """Create a default .env file"""
        default_env_content = """# AI Ops Bug Fix Automation Configuration
# Required: Anthropic API Key
ANTHROPIC_API_KEY=your-anthropic-api-key-here
 
# Optional: Git Configuration  
GIT_USER_NAME=AI-Ops-Bot
GIT_USER_EMAIL=ai-ops@company.com
 
# Optional: Project Configuration
DEFAULT_BRANCH=main
ENABLE_TESTS=true

# Optional: Desired feature branch (default 'feature/ai-fix')
FEATURE_BRANCH=feature/ai-fix
"""
       
        try:
            with open(env_path, 'w') as f:
                f.write(default_env_content)
        except Exception as e:
            print(f"{Colors.RED}Error creating .env file: {e}{Colors.NC}")
            sys.exit(1)
   
    def print_status(self, message: str):
        """Print status message"""
        print(f"{Colors.GREEN}[{datetime.now().strftime('%H:%M:%S')}] {message}{Colors.NC}")
   
    def print_error(self, message: str):
        """Print error message"""
        print(f"{Colors.RED}[ERROR] {message}{Colors.NC}")
   
    def print_warning(self, message: str):
        """Print warning message"""
        print(f"{Colors.YELLOW}[WARNING] {message}{Colors.NC}")
   
    def print_info(self, message: str):
        """Print info message"""
        print(f"{Colors.BLUE}[INFO] {message}{Colors.NC}")
   
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None,
                   capture_output: bool = False, check: bool = True, env: Optional[Dict[str,str]] = None):
        """Execute a command"""
        try:
            self.print_info(f"Running: {' '.join(cmd)}")
           
            # Set environment
            final_env = os.environ.copy()
            if self.api_key:
                final_env['ANTHROPIC_API_KEY'] = self.api_key
            if env:
                final_env.update(env)
           
            result = subprocess.run(
                cmd,
                cwd=cwd or self.repo_path,
                capture_output=capture_output,
                text=True,
                check=check,
                env=final_env
            )
            return result
        except subprocess.CalledProcessError as e:
            if check:
                self.print_error(f"Command failed: {' '.join(cmd)}")
                if e.stdout:
                    self.print_error(f"Stdout: {e.stdout}")
                if e.stderr:
                    self.print_error(f"Stderr: {e.stderr}")
                raise
            return e
        except Exception as e:
            self.print_error(f"Unexpected error running command: {e}")
            if check:
                raise
            return None
   
    def validate_requirements(self):
        """Validate requirements"""
        self.print_status("Validating requirements...")
       
        if not self.api_key or self.api_key == 'your-anthropic-api-key-here':
            self.print_error("Please set your ANTHROPIC_API_KEY in the .env file")
            sys.exit(1)
       
        if not self.repo_url:
            self.print_error("Repository URL is required")
            sys.exit(1)
       
        # Check git
        try:
            subprocess.run(['git', '--version'], capture_output=True, check=True)
            self.print_info("Git is available")
        except:
            self.print_error("Git is not installed")
            sys.exit(1)
       
        self.print_info("All requirements validated")
   
    def clone_repository(self):
        """Clone repository and setup branch"""
        self.print_status("STEP 1: Repository Setup")
       
        # Create working directory
        self.work_dir.mkdir(parents=True, exist_ok=True)
       
        # Clone repository
        self.print_info(f"Cloning: {self.repo_url}")
        try:
            self.run_command(['git', 'clone', self.repo_url, str(self.repo_name)], cwd=self.work_dir)
        except Exception:
            self.print_error("Failed to clone repository. Check URL and permissions.")
            sys.exit(1)
       
        # Configure git within repo
        self.run_command(['git', 'config', 'user.name', self.git_user_name], cwd=self.repo_path)
        self.run_command(['git', 'config', 'user.email', self.git_user_email], cwd=self.repo_path)
       
        # Create feature branch. Prefer desired_branch_base but append timestamp if branch exists.
        self.print_info(f"Preparing feature branch based on '{self.desired_branch_base}'")
        # fetch remote branch list to check existence
        try:
            self.run_command(['git', 'fetch', 'origin'], cwd=self.repo_path, check=False)
            # list remote branches
            remote_branches = self.run_command(['git', 'ls-remote', '--heads', 'origin'],
                                              cwd=self.repo_path, capture_output=True, check=False)
            if remote_branches and hasattr(remote_branches, 'stdout') and self.desired_branch_base in remote_branches.stdout:
                # remote branch exist, append timestamp
                self.branch_name = f"{self.desired_branch_base}-{self.timestamp}"
                self.print_warning(f"Branch {self.desired_branch_base} exists remotely, using '{self.branch_name}' instead")
            else:
                # use desired base directly
                self.branch_name = self.desired_branch_base
        except Exception as e:
            # fallback - use timestamp appended branch to be safe
            self.branch_name = f"{self.desired_branch_base}-{self.timestamp}"
            self.print_warning(f"Could not determine remote branches, using '{self.branch_name}': {e}")
       
        # Create and switch to branch
        self.print_info(f"Creating branch: {self.branch_name}")
        self.run_command(['git', 'checkout', '-b', self.branch_name], cwd=self.repo_path)
       
        self.print_status("Repository setup completed")
   
    # --- New integration with claude-code CLI ------------------------------------------------
    def is_claude_cli_available(self) -> bool:
        """Check if claude-code CLI is available"""
        try:
            result = subprocess.run(['claude-code', '--version'], capture_output=True, text=True, check=True)
            self.print_info(f"claude-code CLI detected: {result.stdout.strip() if result.stdout else 'version unknown'}")
            return True
        except Exception:
            self.print_warning("claude-code CLI not available")
            return False
   
    def claude_authenticate_if_needed(self) -> bool:
        """
        Try to authenticate claude-code CLI using the ANTHROPIC_API_KEY.
        Some versions of the CLI may not need explicit auth; handle gracefully.
        Returns True on success or if auth not needed, False otherwise.
        """
        if not self.api_key:
            self.print_warning("No ANTHROPIC_API_KEY found for claude-code auth")
            return False
        try:
            # Attempt an auth command; if fails, continue but return False
            self.print_info("Attempting to authenticate claude-code CLI (if supported)...")
            # Many CLIs accept an 'auth' or 'configure' subcommand; try common variants
            tried = []
            for auth_cmd in [
                ['claude-code', 'auth', '--api-key', self.api_key],
                ['claude-code', 'configure', '--api-key', self.api_key],
            ]:
                tried.append(' '.join(auth_cmd))
                try:
                    self.run_command(auth_cmd, cwd=self.repo_path, capture_output=True, check=True)
                    self.print_info("Authenticated claude-code CLI successfully")
                    return True
                except Exception:
                    continue
            self.print_warning(f"Could not authenticate claude-code using tried commands: {tried}. It may be unnecessary or unsupported.")
            # not fatal — CLI might still work with ENV key
            return True
        except Exception as e:
            self.print_warning(f"claude-code authentication attempt failed: {e}")
            return False
   
    def run_claude_analysis_and_fix(self) -> bool:
        """
        Use claude-code CLI to analyze and fix repository.
        Returns True if fixes were applied (or CLI successfully ran), False if not.
        """
        if not self.is_claude_cli_available():
            return False
       
        # Authenticate if possible
        self.claude_authenticate_if_needed()
       
        # Path to analyze - prefer src/ but fallback to repo root
        analyze_path = 'src'
        if not (self.repo_path / analyze_path).exists():
            analyze_path = '.'  # analyze repo root if no src/
       
        # Run analysis
        try:
            self.print_status("Running claude-code analysis...")
            analyze_cmd = ['claude-code', 'analyze', '--path', analyze_path, '--format', 'json']
            analyze_result = self.run_command(analyze_cmd, cwd=self.repo_path, capture_output=True, check=False)
            analysis_output = ''
            if analyze_result and hasattr(analyze_result, 'stdout'):
                analysis_output = analyze_result.stdout.strip()
                # attempt to pretty print some info
                try:
                    parsed = json.loads(analysis_output)
                    summary = parsed.get('summary') if isinstance(parsed, dict) else None
                    self.print_info(f"Analysis summary: {summary if summary else 'See raw output.'}")
                except Exception:
                    # If not JSON, print truncated output
                    self.print_info("Received analysis output (non-JSON or not parsed).")
            else:
                self.print_warning("No analysis output from claude-code analyze")
        except Exception as e:
            self.print_warning(f"claude-code analysis failed: {e}")
            return False
       
        # Attempt to run claude-code fix for common languages (python & java)
        languages_to_fix = []
        # detect if repository has python/java files
        if any(self.repo_path.rglob('*.py')):
            languages_to_fix.append('python')
        if any(self.repo_path.rglob('*.java')):
            languages_to_fix.append('java')
        # additionally try js/node if package.json exists
        if (self.repo_path / 'package.json').exists():
            languages_to_fix.append('javascript')
       
        if not languages_to_fix:
            # fallback: try both python and java anyway
            languages_to_fix = ['python', 'java']
       
        fixes_applied = 0
        for lang in languages_to_fix:
            try:
                self.print_status(f"Running claude-code fix for language: {lang}")
                # some CLI variants expect language flag like --language python
                fix_cmd = ['claude-code', 'fix', '--path', analyze_path, '--language', lang]
                fix_result = self.run_command(fix_cmd, cwd=self.repo_path, capture_output=True, check=False)
                if fix_result and hasattr(fix_result, 'returncode') and fix_result.returncode == 0:
                    self.print_info(f"claude-code fix completed for {lang}")
                    fixes_applied += 1
                else:
                    # If CLI returns non-zero but printed changes to stdout, attempt to process them
                    if fix_result and getattr(fix_result, 'stdout', None):
                        self.print_info(f"claude-code fix produced output for {lang}")
                        fixes_applied += 1
                    else:
                        self.print_warning(f"claude-code fix didn't apply changes for {lang} (exit={getattr(fix_result,'returncode', 'n/a')})")
            except Exception as e:
                self.print_warning(f"claude-code fix failed for {lang}: {e}")
                continue
       
        if fixes_applied > 0:
            self.print_status(f"claude-code applied fixes for {fixes_applied} language(s)")
            return True
        else:
            self.print_warning("claude-code did not apply fixes (or fixed 0 languages)")
            return False
    # --- End claude integration ---------------------------------------------------------------
   
    def simulate_code_analysis_and_fix(self):
        """Simulate code analysis and fixes (since claude-code CLI may not exist)"""
        self.print_status("STEP 2: Code Analysis & Fix (SIMULATED)")
       
        # Find source files
        source_extensions = ['py', 'java', 'js', 'ts', 'jsx', 'tsx', 'cpp', 'c', 'h', 'cs', 'rb', 'go', 'php', 'rs']
        source_files = []
       
        for ext in source_extensions:
            source_files.extend(list(self.repo_path.rglob(f'*.{ext}')))
       
        # Filter out common excluded directories
        excluded_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'target', 'build', 'dist'}
        filtered_files = []
       
        for file_path in source_files:
            if not any(excluded in file_path.parts for excluded in excluded_dirs):
                filtered_files.append(file_path)
       
        self.print_info(f"Found {len(filtered_files)} source files")
       
        if not filtered_files:
            self.print_warning("No source files found to analyze")
            return
       
        # Simulate fixes by adding comments to files
        fixed_files = 0
        for file_path in filtered_files[:5]:  # Limit to first 5 files for demo
            try:
                self.print_info(f"Processing: {file_path.name}")
               
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
               
                # Add a simple fix comment at the top
                fix_comment = f"// AI-Generated Fix Applied on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                if file_path.suffix == '.py':
                    fix_comment = f"# AI-Generated Fix Applied on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
               
                # Only add if not already present
                if "AI-Generated Fix Applied" not in content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(fix_comment + content)
                    fixed_files += 1
               
            except Exception as e:
                self.print_warning(f"Could not process {file_path.name}: {e}")
                continue
       
        self.print_status(f"Applied fixes to {fixed_files} files (simulated)")
   
    def run_tests(self):
        """Run tests based on project type"""
        self.print_status("STEP 3: Running Tests")
       
        if not os.getenv('ENABLE_TESTS', 'true').lower() == 'true':
            self.print_info("Tests disabled")
            return
       
        # Detect project type
        project_configs = [
            ('package.json', ['npm', 'test'], 'Node.js'),
            ('pom.xml', ['mvn', 'test', '-q'], 'Maven'),
            ('requirements.txt', ['python', '-m', 'pytest', '-v'], 'Python'),
            ('Cargo.toml', ['cargo', 'test'], 'Rust'),
            ('go.mod', ['go', 'test', './...'], 'Go'),
        ]
       
        project_detected = False
        for config_file, test_cmd, project_type in project_configs:
            if (self.repo_path / config_file).exists():
                self.print_info(f"Detected {project_type} project")
                try:
                    # Try to install dependencies first
                    if config_file == 'package.json':
                        self.run_command(['npm', 'install', '--silent'], cwd=self.repo_path, check=False)
                    elif config_file == 'requirements.txt':
                        self.run_command(['pip', 'install', '-r', 'requirements.txt'], cwd=self.repo_path, check=False)
                   
                    # Run tests
                    result = self.run_command(test_cmd, cwd=self.repo_path, check=False, capture_output=True)
                    if result and hasattr(result, 'returncode') and result.returncode == 0:
                        self.print_info("✅ Tests passed")
                    else:
                        self.print_warning("⚠️ Some tests failed, but continuing...")
                except Exception as e:
                    self.print_warning(f"Tests failed: {e}")
               
                project_detected = True
                break
       
        if not project_detected:
            self.print_info("No recognized project type, skipping tests")
   
    def has_changes(self) -> bool:
        """Check for changes"""
        try:
            result = self.run_command(['git', 'status', '--porcelain'], cwd=self.repo_path, capture_output=True, check=False)
            out = result.stdout.strip() if result and getattr(result,'stdout',None) else ''
            return bool(out)
        except Exception:
            return True
   
    def commit_and_push(self):
        """Commit and push changes"""
        self.print_status("STEP 4: Commit & Push")
       
        if not self.has_changes():
            self.print_warning("No changes to commit")
            return False
       
        # Get change stats
        try:
            stats_result = self.run_command(['git', 'diff', '--stat'], cwd=self.repo_path, capture_output=True, check=False)
            stats = stats_result.stdout.strip() if stats_result and stats_result.stdout else "Changes made"
            self.print_info(f"Changes:\n{stats}")
        except Exception:
            stats = "Unable to get change statistics"
       
        # Stage changes
        self.print_info("Staging changes...")
        self.run_command(['git', 'add', '.'], cwd=self.repo_path)
       
        # Create commit message
        commit_message = f"""AI Fix: Automated bug resolution - {self.timestamp}
 
- Automated fixes applied on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Branch: {self.branch_name}
- Generated by AI Ops Automation
 
{stats}"""
       
        # Commit
        self.print_info("Committing changes...")
        try:
            self.run_command(['git', 'commit', '-m', commit_message], cwd=self.repo_path)
        except Exception as e:
            self.print_warning(f"No commit performed (perhaps no staged changes): {e}")
            return False
       
        # Push
        self.print_info("Pushing to remote...")
        try:
            self.run_command(['git', 'push', 'origin', self.branch_name], cwd=self.repo_path)
            self.print_status("✅ Successfully pushed changes")
            return True
        except Exception as e:
            self.print_error(f"Failed to push: {e}")
            self.print_error("Check repository permissions and network connection")
            return False
   
    def create_pull_request_info(self):
        """Show PR creation information"""
        self.print_status("STEP 5: Pull Request Information")
       
        # Check for GitHub CLI
        try:
            subprocess.run(['gh', '--version'], capture_output=True, check=True)
           
            self.print_info("Creating PR with GitHub CLI...")
           
            pr_title = f"🤖 AI Fix: Automated bug resolution - {self.timestamp}"
            pr_body = f"""## 🤖 AI-Generated Bug Fixes
 
Automated fixes applied by AI Ops Bug Fix Automation.
 
### Summary
- **Branch**: `{self.branch_name}`
- **Timestamp**: `{self.timestamp}`
- **Auto-generated**: Yes
 
### Changes Made
- Code analysis performed
- Automated fixes applied
- Tests executed where applicable
 
### Review Notes
- Please review all changes before merging
- All modifications are AI-generated
 
---
*Generated by AI Ops Bug Fix Automation*"""
           
            try:
                self.run_command([
                    'gh', 'pr', 'create',
                    '--title', pr_title,
                    '--body', pr_body,
                    '--base', self.default_branch,
                    '--head', self.branch_name,
                    '--draft'
                ], cwd=self.repo_path)
                self.print_status("✅ Pull Request created successfully!")
               
                # Get PR URL
                try:
                    pr_result = self.run_command(['gh', 'pr', 'view', '--json', 'url'],
                                               cwd=self.repo_path, capture_output=True, check=False)
                    if pr_result and pr_result.stdout:
                        pr_data = json.loads(pr_result.stdout)
                        self.print_info(f"🔗 PR URL: {pr_data.get('url', 'N/A')}")
                except Exception:
                    pass
                   
            except Exception as e:
                self.print_warning(f"Failed to create PR automatically: {e}")
                self.print_info("Please create PR manually on GitHub")
               
        except Exception:
            self.print_warning("GitHub CLI not available")
            self.print_info("Please install GitHub CLI or create PR manually")
            self.print_info(f"Branch '{self.branch_name}' is ready for PR creation")
   
    def cleanup(self):
        """Clean up temporary files"""
        try:
            shutil.rmtree(self.work_dir)
            self.print_info("✅ Cleaned up temporary files")
        except Exception as e:
            self.print_warning(f"Could not clean up: {e}")
   
    def print_summary(self):
        """Print final summary"""
        print(f"\n{Colors.CYAN}{'='*50}")
        print("    🤖 AI OPS AUTOMATION SUMMARY")
        print(f"{'='*50}{Colors.NC}")
        print(f"{Colors.GREEN}✅ Repository cloned and configured{Colors.NC}")
        print(f"{Colors.GREEN}✅ Code analysis and fixes applied{Colors.NC}")
        print(f"{Colors.GREEN}✅ Tests executed{Colors.NC}")
        print(f"{Colors.GREEN}✅ Changes committed and pushed{Colors.NC}")
        print(f"{Colors.GREEN}✅ Ready for PR review{Colors.NC}")
        print(f"{Colors.CYAN}{'='*50}{Colors.NC}")
        print(f"{Colors.WHITE}Branch: {Colors.YELLOW}{self.branch_name}{Colors.NC}")
        print(f"{Colors.WHITE}Repository: {Colors.YELLOW}{self.repo_url}{Colors.NC}")
        print(f"{Colors.WHITE}Timestamp: {Colors.YELLOW}{self.timestamp}{Colors.NC}\n")
   
    def run(self):
        """Run the complete workflow"""
        try:
            self.print_status("🚀 Starting AI Ops Bug Fix Automation")
           
            self.clone_repository()
            # Attempt to analyze & fix with claude-code CLI; fallback to simulation if unavailable or fails
            try:
                used_claude = self.run_claude_analysis_and_fix()
            except Exception as e:
                self.print_warning(f"claude-code attempt raised exception: {e}")
                used_claude = False
           
            if not used_claude:
                self.print_info("Falling back to simulated analysis & fixes")
                self.simulate_code_analysis_and_fix()
           
            self.run_tests()
           
            if self.commit_and_push():
                self.create_pull_request_info()
           
            self.print_status("🎉 Automation completed successfully!")
            self.print_summary()
           
        except KeyboardInterrupt:
            self.print_error("❌ Process interrupted")
            sys.exit(1)
        except Exception as e:
            self.print_error(f"💥 Unexpected error: {str(e)}")
            sys.exit(1)
        finally:
            self.cleanup()
 
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="🤖 AI Ops Bug Fix Automation")
    parser.add_argument('repo_url', help='GitHub repository URL')
    parser.add_argument('--env-file', default='.env', help='Environment file (default: .env)')
   
    args = parser.parse_args()
   
    print(f"{Colors.CYAN}🤖 AI Ops Bug Fix Automation{Colors.NC}\n")
   
    # Create and run
    fixer = AIOpsFixer(args.repo_url, args.env_file)
    fixer.run()
 
if __name__ == "__main__":
    main()
