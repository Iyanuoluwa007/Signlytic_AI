# Turns on "Include microphone audio" in Windows Live Captions.
#
# Live Captions keeps this preference to itself. There is no documented API and
# nothing lands in the registry or in a package LocalState folder, so the only
# way in is UI Automation against its settings menu:
#   Settings (SettingsButton) > Preferences (PreferencesButton)
#     > Include microphone audio (MicrophoneMenuFlyoutItem, Toggle pattern)
#
# Two behaviours of that menu drive the shape of this script:
#
#  1. The flyout light-dismisses as soon as Live Captions is not the foreground
#     window. Opening it while another app has focus appears to work, but the
#     submenu is gone again before the microphone item can be found. So the
#     window is focused first, and the original foreground window is restored
#     afterwards.
#  2. Toggling a menu item dismisses the flyout by itself, so no key needs to be
#     sent in the success path.
#
# This is best effort by design. It runs as its own short-lived process so that
# a failure here, or a menu renamed by a future Windows build, can never take
# the caption reader down with it. It prints a single JSON line and exits.
#
# Exit codes: 0 done (or already on), 2 Live Captions window not found,
# 3 the menu did not expose the expected items.

param(
  # How long to keep retrying while Live Captions finishes starting up.
  [int]$TimeoutMs = 8000
)

$ErrorActionPreference = "Continue"

function Emit($status, $detail) {
  $o = [ordered]@{ type = "microphone"; status = $status; detail = $detail }
  Write-Output ($o | ConvertTo-Json -Compress)
}

try {
  Add-Type -AssemblyName UIAutomationClient -ErrorAction Stop
  Add-Type -AssemblyName UIAutomationTypes -ErrorAction Stop
} catch {
  Emit "error" "UI Automation is not available on this system"
  exit 3
}
try { Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop } catch { }

try {
  Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class SigFg {
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
}
"@ -ErrorAction Stop
} catch { }

$AE = [System.Windows.Automation.AutomationElement]
$DESC = [System.Windows.Automation.TreeScope]::Descendants
$CHILD = [System.Windows.Automation.TreeScope]::Children

# The window element is re-fetched for every lookup rather than cached. Once
# popups have opened and closed a cached reference goes stale, and every later
# FindFirst against it quietly returns nothing, which reads as "the menu item
# does not exist" when in fact it is right there.
function Get-Win {
  try {
    return $AE::RootElement.FindFirst($CHILD,
      (New-Object System.Windows.Automation.PropertyCondition($AE::ClassNameProperty, "LiveCaptionsDesktopWindow")))
  } catch { return $null }
}

function Find-Id($id) {
  $w = Get-Win
  if (-not $w) { return $null }
  try {
    return $w.FindFirst($DESC, (New-Object System.Windows.Automation.PropertyCondition($AE::AutomationIdProperty, $id)))
  } catch { return $null }
}

function Get-ToggleState($el) {
  try {
    $p = $null
    if ($el.TryGetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern, [ref]$p)) {
      return $p.Current.ToggleState.ToString()
    }
  } catch { }
  return $null
}

function Send-Escape {
  try { [System.Windows.Forms.SendKeys]::SendWait("{ESC}") } catch { }
}

# Wait for Live Captions to have a window, which it will not on a cold start.
$deadline = (Get-Date).AddMilliseconds($TimeoutMs)
$win = $null
while ((Get-Date) -lt $deadline) {
  $win = Get-Win
  if ($win) { break }
  Start-Sleep -Milliseconds 300
}
if (-not $win) {
  Emit "error" "Live Captions window not found"
  exit 2
}

# Give focus back to whatever the user was in when this finishes.
$previousFg = [IntPtr]::Zero
try { $previousFg = [SigFg]::GetForegroundWindow() } catch { }

function Restore-Foreground {
  if ($previousFg -ne [IntPtr]::Zero) {
    try { [SigFg]::SetForegroundWindow($previousFg) | Out-Null } catch { }
  }
}

# One full attempt at open > expand > read the item. The flyout can dismiss
# under us, so this is retried rather than assumed to work first time.
function Try-Once {
  try { (Get-Win).SetFocus() } catch { }
  Start-Sleep -Milliseconds 350

  $btn = Find-Id "SettingsButton"
  if (-not $btn) { return $null }
  try {
    $btn.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
  } catch { return $null }
  Start-Sleep -Milliseconds 700

  $prefs = Find-Id "PreferencesButton"
  if (-not $prefs) { return $null }
  try {
    $prefs.GetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern).Expand()
  } catch { return $null }

  # The submenu is built asynchronously, so poll instead of sleeping a fixed
  # amount and hoping it arrived.
  $sub = (Get-Date).AddMilliseconds(2500)
  $item = $null
  while ((Get-Date) -lt $sub -and -not $item) {
    Start-Sleep -Milliseconds 150
    $item = Find-Id "MicrophoneMenuFlyoutItem"
  }
  return $item
}

$item = $null
for ($attempt = 1; $attempt -le 3 -and -not $item; $attempt++) {
  $item = Try-Once
  if (-not $item) {
    # Clear anything half-open before trying again. Live Captions is focused at
    # this point, so the key lands on its menu and not on the user's app.
    Send-Escape
    Start-Sleep -Milliseconds 400
    Send-Escape
    Start-Sleep -Milliseconds 400
  }
}

if (-not $item) {
  Restore-Foreground
  Emit "error" "microphone option not found in this build of Live Captions"
  exit 3
}

# The state has to be read while the flyout is still open. Once it closes the
# item leaves the tree entirely, and a later read returns nothing, which is not
# the same as the option being off.
$before = Get-ToggleState $item

if ($before -eq "On") {
  # Nothing was clicked, so the menu is still sitting open. Live Captions has
  # focus at this point, so the key lands on its menu rather than on whatever
  # the user was watching.
  Send-Escape
  Start-Sleep -Milliseconds 250
  Send-Escape
  Restore-Foreground
  Emit "ok" "microphone audio was already on"
  exit 0
}

if ($before -ne "Off") {
  Send-Escape
  Start-Sleep -Milliseconds 250
  Send-Escape
  Restore-Foreground
  Emit "error" "could not read the microphone option"
  exit 3
}

try {
  $item.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern).Toggle()
} catch {
  Send-Escape
  Restore-Foreground
  Emit "error" "could not switch the microphone option on"
  exit 3
}
Start-Sleep -Milliseconds 700

# Toggling dismisses the flyout, so confirm by opening it again and reading the
# value rather than assuming the click did what was intended.
$verify = Try-Once
$after = $null
if ($verify) {
  $after = Get-ToggleState $verify
  Send-Escape
  Start-Sleep -Milliseconds 250
  Send-Escape
}
Restore-Foreground

if ($after -eq "On") { Emit "ok" "microphone audio switched on"; exit 0 }
if (-not $after) {
  # The toggle was invoked and did not throw, but the menu could not be
  # reopened to confirm it. Say so rather than claiming success.
  Emit "ok" "microphone audio switched on (unverified)"
  exit 0
}

Emit "error" ("microphone option still reads " + $after)
exit 3
