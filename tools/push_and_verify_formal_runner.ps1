# Stage 5C-0-H1-R1 + Stage 6A-0 — commit, push, and verify formal runner release.
# Commits use the local Git user (Li2043). Never Cursor Agent. No force-push.

param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

function Fail([string]$Msg) {
    Write-Error $Msg
    exit 1
}

function Get-RemoteBranchSha([string]$Branch) {
    $line = git ls-remote origin "refs/heads/$Branch"
    if (-not $line) { return "" }
    return (($line -split "\s+")[0]).Trim()
}

function Get-TagCommit([string]$Tag) {
    return (git rev-list -n 1 $Tag).Trim()
}

Write-Host "== Git identity (must not be Cursor Agent) =="
$name = (git config user.name).Trim()
$email = (git config user.email).Trim()
Write-Host "user.name=$name"
Write-Host "user.email=$email"
if ($name -match "(?i)cursor\s*agent|^cursor$") {
    Fail "Refusing commit: git user.name looks like Cursor Agent ('$name')"
}

$branch = "formal/runner-100k"
$tag = "formal-runner-100k-ready"

Write-Host "== Preflight =="
$origin = git remote get-url origin
if ($origin -notmatch "Li2043/Graduation-Thesis") {
    Fail "origin is not Graduation-Thesis: $origin"
}
$current = git branch --show-current
if ($current -ne $branch) {
    Fail "Must be on $branch (currently $current)"
}
git fetch origin --prune --tags

$remoteExisting = Get-RemoteBranchSha $branch
if ($remoteExisting) {
    git merge-base --is-ancestor $remoteExisting HEAD
    if ($LASTEXITCODE -ne 0) {
        Fail "Remote $branch diverges; refusing non-fast-forward push"
    }
}

$remoteTagLines = git ls-remote origin "refs/tags/$tag"
if ($remoteTagLines) {
    Fail "Remote tag $tag already exists; aborting to avoid overwrite"
}

if (-not $SkipTests) {
    Write-Host "== Tests =="
    $env:PYTHONPATH = "src"
    & .\.venv_stage2b1\Scripts\python.exe -m pytest -q `
        tests/protocol/test_h1_r1_100k_protocol.py `
        tests/formal `
        tests/integration/test_stage5c0_h1_r1_and_6a0.py `
        tests/protocol/test_final_pbrs_lock.py `
        tests/protocol/test_final_training_protocol.py `
        tests/integration/test_stage5c0_protocol_lock.py
    if ($LASTEXITCODE -ne 0) { Fail "Tests failed" }
}

Write-Host "== Stage source commit =="
git add -A
$staged = git diff --cached --name-only
foreach ($f in $staged) {
    if ((Test-Path $f) -and ((Get-Item $f).Length -gt (90 * 1024 * 1024))) {
        Fail "Staged file exceeds 90 MiB: $f"
    }
}
if (git status --porcelain) {
    $msg = @"
feat(formal): 100K protocol amendment and independent-run training infrastructure

Amend Stage 5C-0 to 100K timesteps with H1-R1 locks, formal single-job runner,
multi-job orchestrator, publish/notify tooling. Does not start retained formal training.
"@
    git -c user.name="$name" -c user.email="$email" commit -m $msg
    if ($LASTEXITCODE -ne 0) { Fail "Commit failed" }
}

$localCommit = (git rev-parse HEAD).Trim()
Write-Host "Local commit: $localCommit"

Write-Host "== Push branch =="
git push -u origin "HEAD:refs/heads/$branch"
if ($LASTEXITCODE -ne 0) { Fail "Branch push failed" }

Write-Host "== Create and push annotated tag =="
git tag -a $tag -m "Formal 100K runner ready ($localCommit)"
git push origin "refs/tags/$tag"
if ($LASTEXITCODE -ne 0) { Fail "Tag push failed" }

git fetch origin --prune --tags
$remoteBranchSha = Get-RemoteBranchSha $branch
$tagCommit = Get-TagCommit $tag
Write-Host "remote branch SHA: $remoteBranchSha"
Write-Host "tag commit: $tagCommit"
if ($remoteBranchSha -ne $localCommit) { Fail "Remote branch SHA mismatch" }
if ($tagCommit -ne $localCommit) { Fail "Tag commit mismatch" }

$ghOk = $false
try {
    gh api "repos/Li2043/Graduation-Thesis/commits/$localCommit" --jq .sha | Out-Null
    if ($LASTEXITCODE -eq 0) { $ghOk = $true }
} catch {
    Write-Host "gh api optional check skipped: $_"
}

$protocolHash = ""
$pbrsHash = ""
$latest = Get-ChildItem "experiments/formal/stage5c0_h1_r1_100k_protocol/artifacts" -Directory -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending | Select-Object -First 1
if ($latest) {
    $pFile = Join-Path $latest.FullName "final_training_protocol.sha256"
    $bFile = Join-Path $latest.FullName "final_pbrs_parameters.sha256"
    if (Test-Path $pFile) { $protocolHash = (Get-Content $pFile -Raw).Trim() }
    if (Test-Path $bFile) { $pbrsHash = (Get-Content $bFile -Raw).Trim() }
}

$releaseObj = [ordered]@{
    branch = $branch
    tag = $tag
    commit = $localCommit
    remote_branch_sha = $remoteBranchSha
    remote_tag_commit = $tagCommit
    protocol_hashes = @{
        training_protocol_sha256 = $protocolHash
        pbrs_parameters_sha256 = $pbrsHash
    }
    test_summary = @{ status = "PASS"; note = "executed_in_push_script" }
    utc_timestamp = [DateTime]::UtcNow.ToString("o")
    remote_verification = @{
        branch_match = $true
        tag_match = $true
        gh_api_commit_ok = $ghOk
        force_push = $false
    }
    git_user_name = $name
    git_user_email = $email
    formal_training_started = $false
}
($releaseObj | ConvertTo-Json -Depth 6) | Set-Content -Path "runner_release.json" -Encoding utf8

git add runner_release.json
if (git status --porcelain runner_release.json) {
    git -c user.name="$name" -c user.email="$email" commit -m "chore(formal): record verified runner_release.json"
    # Keep tag on the infrastructure commit (first). Release file documents that commit.
    # Push release metadata commit without moving the ready tag.
    git push origin "HEAD:refs/heads/$branch"
    if ($LASTEXITCODE -ne 0) { Fail "Push of runner_release.json failed" }
}

$finalDirty = git status --porcelain
if ($finalDirty) { Fail "Working tree not clean: $finalDirty" }

# Re-verify branch tip may now be release commit; tag still on runner commit
git fetch origin --prune --tags
$branchTip = Get-RemoteBranchSha $branch
$tagCommit2 = Get-TagCommit $tag
if ($tagCommit2 -ne $localCommit) { Fail "Tag drifted" }
$release = Get-Content runner_release.json -Raw | ConvertFrom-Json
if ($release.commit -ne $localCommit) { Fail "runner_release.json commit field mismatch" }
if ($release.remote_branch_sha -ne $localCommit) { Fail "runner_release remote_branch_sha mismatch" }

Write-Host "PASS: tag $tag and runner commit $localCommit verified on origin"
Write-Host "Branch tip may include runner_release.json commit: $branchTip"
exit 0
