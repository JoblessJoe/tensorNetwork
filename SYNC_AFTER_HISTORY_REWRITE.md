# Handoff: sync the training rig after the history rewrite (2026-10-08)

**For the user and any agent on the training rig.** Delete this file in the same commit that finishes the sync.

## What happened
- The repo went **public** on 2026-10-08.
- Before that, the history was rewritten with `git filter-repo` to scrub private infra details: the Tailscale IP, the machine name (now `training-pc`) and the sshd password note. All 142 commits were kept, with new SHAs.
- **Old GitHub tip `8d7b336` = new tip `77d10d2`** (identical files except the scrubbed text). On top of it: `README.md` and this file.
- The rig's working tree, uncommitted changes and running training were **not touched**. Its `.git` still has the old history.

## Rules
- **Never** `git reset --hard`, `git checkout -- .`, `git clean` or `git stash drop` here. The uncommitted work is not on GitHub.
- **Never** `git push --force` from the rig. It would put the old, unscrubbed history back on the public repo.
- **Public repo from now on:** no IPs, hostnames, usernames, ssh/Tailscale details or secrets in tracked files (AGENTS.md, HANDOFF.md, logs). Write "the training PC" instead.

## Steps (safe while training runs: nothing below changes the working tree until step 5)

1. **Back up the folder.** `cp -r <repo> <repo>-backup-2026-10-08`
2. **Check where HEAD is.**
   ```bash
   git rev-parse --short HEAD            # expect 8d7b336
   git status --short | head -50
   ```
   - **HEAD is `8d7b336`** → go on with step 3.
   - **Anything else** (local commits on top, or an older commit) → **stop** and ask in chat. Show the output of `git log --oneline -5` and `git log --oneline HEAD --not 8d7b336`.
3. **Fetch.** `git fetch origin`. This changes no files.
4. **Move HEAD to the new history without touching any file.**
   ```bash
   git reset --mixed origin/main         # moves HEAD + index only; working tree stays as is
   git restore README.md SYNC_AFTER_HISTORY_REWRITE.md   # these two only exist on the new side
   ```
5. **Scrub the working tree.** The uncommitted AGENTS.md / HANDOFF.md still have the old lines. Replace them the same way the rewrite did:
   ```bash
   grep -rnE "100\.116\.105\.116|wsl-pc-jo|PasswordAuthentication" --exclude-dir=venv --exclude-dir=venv-p40 --exclude-dir=.git --exclude-dir=models .
   ```
   - `100.116.105.116` → `[tailscale-ip]`
   - `` `wsl-pc-jo` `` / `wsl-pc-jo` → `` `training-pc` `` / `training-pc`
   - delete ``(key login; `PasswordAuthentication no` still recommended)`` (with the leading space)
   - Re-run the grep. It must print nothing.
6. **Check the diff.** `git status` and `git diff --stat` should show only the rig's own uncommitted work. Nothing should appear as deleted that the rig didn't delete.
7. **Commit and push normally.** Ask the user what to commit first (models/ is large). Delete this file in that commit. Then a plain `git push` (no `--force`).
8. **Once it all looks right,** delete the backup folder.

## Other clones
- `~/projects/NeuralNetworks/tensorNetwork` on the Linux box (Claude Code side) has no own work. Easiest: re-clone it.
