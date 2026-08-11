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
# Output lines:
#   {"type":"status","state":"waiting|attached|idle","detail":"..."}
#   {"type":"caption","text":"...","ts":1234567890}
#   {"type":"error","message":"..."}

[CmdletBinding()]
param(
    # How often to sample the caption element, in milliseconds.
    [int]$PollMs = 400,
    # Give up looking for the window after this many seconds (0 = wait forever).
    [int]$AttachTimeoutSec = 0
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Write-Line($obj) {
    # Compress so each record stays on a single line for the reader
    [Console]::Out.WriteLine(($obj | ConvertTo-Json -Compress -Depth 4))
    [Console]::Out.Flush()
}

try {
    Add-Type -AssemblyName UIAutomationClient
    Add-Type -AssemblyName UIAutomationTypes
} catch {
    Write-Line @{ type = "error"; message = "UI Automation unavailable: $($_.Exception.Message)" }
    exit 1
}

$WINDOW_CLASS = "LiveCaptionsDesktopWindow"
$CAPTION_ID = "CaptionsTextBlock"
# Shown while Live Captions is open but has not heard anything yet
$IDLE_ID = "ReadyToCaptionTextBlock"

$root = [System.Windows.Automation.AutomationElement]::RootElement

function Find-CaptionWindow {
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ClassNameProperty, $WINDOW_CLASS)
    return $root.FindFirst([System.Windows.Automation.TreeScope]::Children, $cond)
}

function Find-CaptionElement($win, $automationId) {
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::AutomationIdProperty, $automationId)
    return $win.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $cond)
}

Write-Line @{ type = "status"; state = "waiting"; detail = "looking for the Live Captions window" }

$deadline = if ($AttachTimeoutSec -gt 0) { (Get-Date).AddSeconds($AttachTimeoutSec) } else { [DateTime]::MaxValue }
$win = $null
while (-not $win) {
    $win = Find-CaptionWindow
    if ($win) { break }
    if ((Get-Date) -gt $deadline) {
        Write-Line @{ type = "error"; message = "Live Captions window not found" }
        exit 2
    }
    Start-Sleep -Milliseconds 700
}

Write-Line @{ type = "status"; state = "attached"; detail = "reading captions" }

$last = ""
$wasIdle = $false

while ($true) {
    try {
        $el = Find-CaptionElement $win $CAPTION_ID
        if (-not $el) {
            # No caption element yet. If the idle placeholder is present the
            # window is alive and simply has not heard speech; otherwise the
            # window has gone and we should re-attach.
            $idle = Find-CaptionElement $win $IDLE_ID
            if ($idle) {
                if (-not $wasIdle) {
                    Write-Line @{ type = "status"; state = "idle"; detail = $idle.Current.Name }
                    $wasIdle = $true
                }
            } else {
                $probe = Find-CaptionWindow
                if (-not $probe) {
                    Write-Line @{ type = "status"; state = "waiting"; detail = "Live Captions window closed" }
                    $win = $null
                    while (-not $win) { Start-Sleep -Milliseconds 700; $win = Find-CaptionWindow }
                    Write-Line @{ type = "status"; state = "attached"; detail = "reattached" }
                    $last = ""
                } else {
                    $win = $probe
                }
            }
            Start-Sleep -Milliseconds $PollMs
            continue
        }

        $wasIdle = $false
        $text = $el.Current.Name
        if ($text -and $text -ne $last) {
            $last = $text
            Write-Line @{ type = "caption"; text = $text; ts = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() }
        }
    } catch {
        # A transient UIA failure (window redrawing, element swapped out) should
        # not kill the sidecar; drop the cached window and let the loop recover.
        $win = Find-CaptionWindow
        if (-not $win) {
            Write-Line @{ type = "status"; state = "waiting"; detail = "lost the window, retrying" }
            while (-not $win) { Start-Sleep -Milliseconds 700; $win = Find-CaptionWindow }
            Write-Line @{ type = "status"; state = "attached"; detail = "reattached" }
            $last = ""
        }
    }

    Start-Sleep -Milliseconds $PollMs
}
