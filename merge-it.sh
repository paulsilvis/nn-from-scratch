#!/bin/bash
set -e

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <bundle-filename-in-~/Downloads> <branch-name>"
  echo "Example: $0 ch5-perceptron.bundle claude-ch5-perceptron"
  exit 1
fi

BUNDLE="$HOME/Downloads/$1"
BRANCH="$2"

if [ ! -f "$BUNDLE" ]; then
  echo "Bundle not found: $BUNDLE"
  exit 1
fi

cd ~/neural-networks/nn-from-scratch
git fetch "$BUNDLE" "$BRANCH:$BRANCH"
git log --oneline "$BRANCH" -3
git checkout main
git merge "$BRANCH"
git push
git branch -d "$BRANCH"
rm "$BUNDLE"

