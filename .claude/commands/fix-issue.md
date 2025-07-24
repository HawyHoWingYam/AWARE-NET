# Fix Issue Command

Analyze a GitHub issue and create a comprehensive fix with pull request.

## Parameters
- $1: Issue number (required)
- $2: Branch name prefix (optional, defaults to "fix")

## Execution Steps

### 1. Context Gathering
```bash
# Get issue details
gh issue view $1 --json title,body,labels,comments --web

# Search for related code
gh search code --repo HawyHoWingYam/MobileDeepfakeDetection --query "$KEYWORDS"

# Check recent related PRs
gh pr list --search "involves:$ISSUE_AUTHOR" --state all
```

### 2. Issue Analysis
- Analyze the issue description and comments
- Identify root cause based on error messages/symptoms
- Determine affected components (Stage 1, Stage 2, preprocessing, etc.)
- Classify issue severity and complexity

### 3. Solution Planning
- Design fix approach based on issue analysis
- Consider backward compatibility requirements
- Plan testing strategy
- Identify documentation updates needed

### 4. Implementation
- Create feature branch: `$2/issue-$1-brief-description`
- Implement the fix with proper error handling
- Add/update unit tests for the fix
- Update documentation if needed

### 5. Quality Assurance
- Run relevant test suites
- Perform code quality checks
- Test on appropriate hardware (CPU/GPU)
- Verify fix doesn't introduce regressions

### 6. Pull Request Creation
- Create PR with descriptive title: `[Fix] Brief description (fixes #$1)`
- Use standard PR template with all sections filled
- Link to original issue
- Add appropriate labels based on component affected

### 7. Follow-up
- Monitor CI/CD pipeline status
- Respond to review comments
- Update PR based on feedback
- Ensure issue is closed when PR is merged

## Expected Outcome
- Issue is thoroughly analyzed with root cause identified
- Comprehensive fix is implemented with tests
- PR is created following project standards
- All quality gates pass
- Issue reporter is satisfied with resolution

## Usage Examples
```bash
# Fix issue #123 with default branch naming
@claude /fix-issue 123

# Fix issue #456 with custom branch prefix
@claude /fix-issue 456 hotfix

# Complex issue requiring multiple components
@claude /fix-issue 789 feature
```

## Success Criteria
- [ ] Issue root cause clearly identified
- [ ] Fix addresses the core problem
- [ ] No regressions introduced
- [ ] Appropriate tests added/updated
- [ ] Documentation updated if needed
- [ ] PR follows project conventions
- [ ] All CI/CD checks pass