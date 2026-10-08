#!/usr/bin/env bash
# Waits until the running nightRun.py has finished (incl. its final benchmark), then commits and pushes everything in both repos
# (tensorNetwork and ../sprudelJump). Retries a failed push a few times. Does NOT shut anything down.
#
#   setsid nohup bash commitAfterRun.sh > models/autocommit_log.txt 2>&1 &
#   DRY_RUN=1 SKIP_WAIT=1 bash commitAfterRun.sh      test: shows what would be committed, changes nothing
cd "$(dirname "$0")" || exit 1
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [ -z "$SKIP_WAIT" ]; then
    say "waiting for nightRun.py to finish ..."
    while pgrep -f "python.*nightRun.py" > /dev/null; do sleep 60; done
    sleep 20                                    # let the last files be written
fi
last=$(grep -E "run done|FAILED|benchmark written" models/night_log.txt | tail -1 | cut -c1-80)
say "nightRun.py finished. Last log line: ${last}"

commitRepo() {                                  # $1 = repo directory, $2 = commit message
    cd "$1" || return 1
    git add -A . ':(exclude)__pycache__' > /dev/null 2>&1
    if git diff --cached --quiet; then say "$1: nothing to commit"; return 0; fi
    if [ -n "$DRY_RUN" ]; then say "$1: would commit:"; git diff --cached --stat | tail -5; git reset -q; return 0; fi
    git commit -q -m "$2" || { say "$1: commit failed"; return 1; }
    for attempt in 1 2 3 4 5; do
        if git push -q 2>&1; then say "$1: pushed ($(git log --oneline -1))"; return 0; fi
        say "$1: push failed (attempt $attempt), retrying in 5 min"; sleep 300
    done
    say "$1: PUSH FAILED after 5 attempts - the commit is local only"
    return 1
}

msg="Auto-commit after nightRun: results, models and logs of the finished run ($(date '+%Y-%m-%d %H:%M'))"
here=$(pwd)
commitRepo "$here" "$msg"
cd "$here" && [ -d ../sprudelJump ] && commitRepo "$(cd ../sprudelJump && pwd)" "$msg"
say "done"
