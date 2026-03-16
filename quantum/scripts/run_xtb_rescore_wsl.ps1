param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,
    [string]$Distro = "Ubuntu",
    [string]$XtbBin = "xtb",
    [int]$Charge = 0,
    [int]$Uhf = 0,
    [ValidateSet(0,1,2)]
    [int]$Gfn = 2,
    [string]$Alpb = "",
    [switch]$Opt,
    [double]$StrainWeight = 0.5,
    [switch]$KeepWorkdir
)

$ErrorActionPreference = "Stop"

$repoWindows = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$inputWindows = (Resolve-Path $InputPath).Path

function Convert-ToWslPath {
    param([Parameter(Mandatory = $true)][string]$WindowsPath)
    $normalized = $WindowsPath -replace '\\', '/'
    if ($normalized -match '^([A-Za-z]):/(.*)$') {
        $drive = $matches[1].ToLower()
        $rest = $matches[2]
        return "/mnt/$drive/$rest"
    }
    throw "Unsupported Windows path: $WindowsPath"
}

$repoWsl = Convert-ToWslPath -WindowsPath $repoWindows
$inputWsl = Convert-ToWslPath -WindowsPath $inputWindows

$cmdParts = @(
    "cd '$repoWsl' && python3 quantum/scripts/qm_rescore_xtb.py",
    "--input '$inputWsl'",
    "--xtb_bin '$XtbBin'",
    "--charge $Charge",
    "--uhf $Uhf",
    "--gfn $Gfn",
    "--strain_weight $StrainWeight"
)

if ($Alpb) {
    $cmdParts += "--alpb '$Alpb'"
}
if ($Opt) {
    $cmdParts += "--opt"
}
if ($KeepWorkdir) {
    $cmdParts += "--keep_workdir"
}

$bashCommand = ($cmdParts -join " ")
Write-Host "Running in WSL:" $bashCommand
wsl -d $Distro bash -lc $bashCommand
