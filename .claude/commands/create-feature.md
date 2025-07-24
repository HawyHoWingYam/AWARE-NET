# Create Feature Command

Create a new feature branch and implement basic feature structure.

## Parameters
- $1: Feature description (required) - Brief description of the feature
- $2: Component ("stage1", "stage2", "preprocessing", "evaluation", "deployment") - required
- $3: Priority ("low", "medium", "high", "critical") - optional, defaults to "medium"

## Execution Steps

### 1. Feature Analysis
- Parse feature description for key requirements
- Identify affected components and files
- Determine implementation complexity
- Plan feature architecture

### 2. Branch Creation
```bash
# Create feature branch with descriptive name
BRANCH_NAME="feature/${COMPONENT}-$(echo $1 | tr ' ' '-' | tr '[:upper:]' '[:lower:]')"
git checkout -b $BRANCH_NAME

# Push branch to remote
git push -u origin $BRANCH_NAME
```

### 3. Issue Creation
```bash
# Create corresponding GitHub issue
gh issue create \
  --title "Feature: $1" \
  --body "$(cat <<EOF
## Feature Description
$1

## Component
$2

## Priority
$3

## Implementation Plan
- [ ] Design feature architecture
- [ ] Implement core functionality
- [ ] Add comprehensive tests
- [ ] Update documentation
- [ ] Performance validation

## Acceptance Criteria
- [ ] Feature works as described
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Performance meets requirements
- [ ] Code follows project standards

This issue was automatically created by Claude Code.
EOF
)" \
  --label "feature,$2,auto-created" \
  --assignee "@me"
```

### 4. Basic Structure Creation

#### For Stage 1 Features:
```python
# Create/update relevant files in src/stage1/
- Feature implementation in appropriate module
- Unit tests in test/
- Configuration updates if needed
```

#### For Stage 2 Features:
```python
# Create/update relevant files in src/stage2/
- EfficientNet enhancements
- GenConViT improvements
- Manager updates
- Integration tests
```

#### For Preprocessing Features:
```python
# Create/update scripts/
- New preprocessing capabilities
- Configuration extensions
- Validation improvements
```

### 5. Feature Implementation Template
Create basic implementation structure:

```python
"""
${FEATURE_NAME} Implementation
===============================

${FEATURE_DESCRIPTION}

Author: Claude Code (Auto-generated)
Created: $(date)
Component: ${COMPONENT}
"""

# Imports
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

# Feature implementation
class ${FEATURE_CLASS_NAME}:
    """
    ${FEATURE_DESCRIPTION}
    
    This class implements the core functionality for ${FEATURE_NAME}.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # TODO: Initialize feature components
        
    def forward(self, x):
        """Forward pass implementation"""
        # TODO: Implement feature logic
        pass
        
    def get_config(self) -> Dict[str, Any]:
        """Return feature configuration"""
        return self.config

# Factory function
def create_${FEATURE_NAME}(config: Optional[Dict[str, Any]] = None):
    """Factory function to create ${FEATURE_NAME} instance"""
    # TODO: Implement factory logic
    pass
```

### 6. Test Structure Creation
```python
"""
Test Suite for ${FEATURE_NAME}
==============================

Comprehensive tests for ${FEATURE_NAME} functionality.
"""

import unittest
import torch
from src.${COMPONENT}.${FEATURE_MODULE} import ${FEATURE_CLASS_NAME}

class Test${FEATURE_CLASS_NAME}(unittest.TestCase):
    """Test cases for ${FEATURE_CLASS_NAME}"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            # TODO: Add test configuration
        }
        self.feature = ${FEATURE_CLASS_NAME}(self.config)
    
    def test_initialization(self):
        """Test feature initialization"""
        # TODO: Implement initialization tests
        pass
    
    def test_forward_pass(self):
        """Test forward pass functionality"""
        # TODO: Implement forward pass tests
        pass
    
    def test_configuration(self):
        """Test configuration management"""
        # TODO: Implement configuration tests
        pass

if __name__ == '__main__':
    unittest.main()
```

### 7. Documentation Updates
- Add feature description to appropriate README sections
- Update CLAUDE.md with new capabilities
- Create usage examples
- Add troubleshooting notes

### 8. Initial Commit
```bash
# Add all new files
git add .

# Create descriptive commit message
git commit -m "feat(${COMPONENT}): Add ${FEATURE_NAME} implementation

- Create basic feature structure
- Add unit test framework  
- Update documentation
- Link to issue #${ISSUE_NUMBER}

This is an initial implementation that provides:
- Core feature architecture
- Basic functionality
- Test framework
- Documentation structure"

# Push to remote branch
git push origin $BRANCH_NAME
```

## Expected Outcomes
- New feature branch created and pushed
- GitHub issue created and linked
- Basic implementation structure in place
- Unit test framework established
- Documentation updated
- Ready for detailed implementation

## Usage Examples
```bash
# Create Stage 2 optimization feature
@claude /create-feature "Multi-precision training support" stage2 high

# Create preprocessing enhancement
@claude /create-feature "Real-time face detection pipeline" preprocessing medium

# Create evaluation feature
@claude /create-feature "Cross-dataset performance analysis" evaluation low
```

## Success Criteria
- [ ] Feature branch created and pushed successfully
- [ ] GitHub issue created with proper labels and description
- [ ] Basic implementation structure is sound
- [ ] Test framework is properly set up
- [ ] Documentation is updated appropriately
- [ ] Initial commit follows project conventions
- [ ] Feature is ready for detailed development

## Related Commands
- `/fix-issue` - For bug fixes
- `/optimize-stage1` - For Stage 1 optimizations  
- `/optimize-stage2` - For Stage 2 optimizations
- `/review-pr` - For code review