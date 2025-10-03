#!/bin/bash

# This script lists all remaining R files that need to be converted to Python
echo "Remaining R files to convert:"
find . -name "*.R" -type f | sort

echo ""
echo "Total R files: $(find . -name "*.R" -type f | wc -l)"
echo "Python files created: $(find . -name "*.py" -type f | wc -l)"
