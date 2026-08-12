# Signlytic Live Captions reader.
#
# Attaches to the Windows 11 Live Captions window via UI Automation and prints
# one JSON object per line to stdout whenever the caption text changes.
# Deliberately dumb: it reports the raw caption buffer and does no sentence
# splitting or de-duplication. That lives in caption-stream.js, because the
# buffer is cumulative and gets retroactively revised, which is far easier to
# reason about in one place.
#
# Uses only assemblies that ship with Windows, so the app needs no .NET SDK,
# no Rust toolchain and no native addon.
#
# Survivability matters more than precision here. UI Automation talks to
# another process over COM, so calls can fail at any moment: the window is
# redrawing, the element was swapped out, Live Captions restarted. None of
# that should kill the reader, so every call site is guarded and the loop
# never lets an error escape to terminate the engine.
#
# Output lines:
#   {"type":"status","state":"waiting|attached|idle","detail":"..."}
#   {"type":"caption","text":"...","ts":1234567890}
#   {"type":"error","message":"..."}

[CmdletBinding()]
param(
    # How often to sample the caption element, in milliseconds.
    [int]$PollMs = 400,
    # Give up looking for the window after this many seconds (0 = wait forever).
    [int]$AttachTimeoutSec = 0,
    # PID of the app that started us. If it dies without closing us down (a
    # crash, or being force-killed), stop rather than lingering as an orphan
    # holding a UI Automation connection. 0 disables the check.
    [int]$ParentPid = 0
)

# Deliberately NOT "Stop": a non-terminating UI Automation hiccup must not
# take the process down. Failures are handled explicitly where they matter.
$ErrorActionPreference = "Continue"
$OutputEncoding = [System.Text.Encoding]::UTF8
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

# Writing to a closed pipe throws. That happens normally when the app exits
# first, so treat it as "we are done" rather than an error.
$script:PipeOpen = $true
function Write-Line($obj) {
    if (-not $script:PipeOpen) { return }
    try {
        [Console]::Out.WriteLine(($obj | ConvertTo-Json -Compress -Depth 4))
        [Console]::Out.Flush()
    } catch {
        $script:PipeOpen = $false
    }
}

try {
    Add-Type -AssemblyName UIAutomationClient -ErrorAction Stop
    Add-Type -AssemblyName UIAutomationTypes -ErrorAction Stop
} catch {
    Write-Line @{ type = "error"; message = "UI Automation unavailable: $($_.Exception.Message)" }
    exit 1
}

$WINDOW_CLASS = "LiveCaptionsDesktopWindow"
$CAPTION_ID = "CaptionsTextBlock"
# Shown while Live Captions is open but has not heard anything yet
$IDLE_ID = "ReadyToCaptionTextBlock"

function Get-Root {
    try { return [System.Windows.Automation.AutomationElement]::RootElement } catch { return $null }
}

function Find-CaptionWindow {
    try {
        $root = Get-Root
        if (-not $root) { return $null }
        $cond = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ClassNameProperty, $WINDOW_CLASS)
        return $root.FindFirst([System.Windows.Automation.TreeScope]::Children, $cond)
    } catch { return $null }
}

function Find-CaptionElement($win, $automationId) {
    if (-not $win) { return $null }
    try {
        $cond = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::AutomationIdProperty, $automationId)
        return $win.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $cond)
    } catch { return $null }
}

function Get-ElementName($el) {
    if (-not $el) { return $null }
    # The element can go stale between being found and being read.
    try { return $el.Current.Name } catch { return $null }
}

Write-Line @{ type = "status"; state = "waiting"; detail = "looking for the Live Captions window" }

$deadline = if ($AttachTimeoutSec -gt 0) { (Get-Date).AddSeconds($AttachTimeoutSec) } else { [DateTime]::MaxValue }

$win = $null
$last = ""
$wasIdle = $false
$attached = $false

while ($script:PipeOpen) {
    if ($ParentPid -gt 0) {
        if (-not (Get-Process -Id $ParentPid -ErrorAction SilentlyContinue)) { break }
    }
    try {
        if (-not $win) {
            $win = Find-CaptionWindow
            if (-not $win) {
                if ((Get-Date) -gt $deadline) {
                    Write-Line @{ type = "error"; message = "Live Captions window not found" }
                    exit 2
                }
                if ($attached) {
                    Write-Line @{ type = "status"; state = "waiting"; detail = "Live Captions window closed" }
                    $attached = $false
                    $last = ""
                }
                Start-Sleep -Milliseconds 700
                continue
            }
            Write-Line @{ type = "status"; state = "attached"; detail = "reading captions" }
            $attached = $true
            $wasIdle = $false
        }

        $el = Find-CaptionElement $win $CAPTION_ID
        if (-not $el) {
            # No caption element. Either the window is alive but idle, or it
            # has gone and we need to re-attach on the next pass.
            $idle = Find-CaptionElement $win $IDLE_ID
            if ($idle) {
                if (-not $wasIdle) {
                    Write-Line @{ type = "status"; state = "idle"; detail = (Get-ElementName $idle) }
                    $wasIdle = $true
                }
            } else {
                $win = $null
            }
            Start-Sleep -Milliseconds $PollMs
            continue
        }

        $wasIdle = $false
        $text = Get-ElementName $el
        if ($null -eq $text) {
            # Element went stale mid-read; drop it and re-resolve next pass.
            $win = $null
            Start-Sleep -Milliseconds $PollMs
            continue
        }
        if ($text -and $text -ne $last) {
            $last = $text
            Write-Line @{ type = "caption"; text = $text; ts = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() }
        }
    } catch {
        # Nothing in the loop is allowed to terminate the process. Forget the
        # cached window and let the next pass re-resolve it.
        $win = $null
        try { Write-Line @{ type = "status"; state = "waiting"; detail = "recovering: $($_.Exception.Message)" } } catch { }
    }

    Start-Sleep -Milliseconds $PollMs
}
