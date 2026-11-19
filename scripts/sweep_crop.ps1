# Compare crop strategies on the SOP validation classes, for pretrained CLIP and a LoRA adapter.
# Run it from an environment where `python` is the one with visual-search installed.
# Usage: scripts\sweep_crop.ps1 -DataDir C:\vs\data -Boxes artifacts\boxes_val.json -Adapter artifacts\m2_clip\best
param(
    [Parameter(Mandatory)] [string] $DataDir,
    [Parameter(Mandatory)] [string] $Boxes,
    [Parameter(Mandatory)] [string] $Adapter,
    [string] $OutDir = "artifacts/m3_sweep"
)

# name, then the crop flags; "full" is the control with no cropping.
$configs = [ordered]@{
    "full"             = @()
    "top_conf25"       = @("--crop-strategy", "top", "--crop-min-confidence", "0.25")
    "top_conf50"       = @("--crop-strategy", "top", "--crop-min-confidence", "0.5")
    "union_conf25"     = @("--crop-strategy", "union", "--crop-min-confidence", "0.25")
    "top_margin25"     = @("--crop-strategy", "top", "--crop-margin", "0.25")
    "top_minarea30"    = @("--crop-strategy", "top", "--crop-min-area", "0.3")
}
$models = [ordered]@{
    "pretrained" = @()
    "lora"       = @("--adapter", $Adapter)
}

New-Item -ItemType Directory -Force $OutDir | Out-Null
foreach ($model in $models.Keys) {
    foreach ($name in $configs.Keys) {
        $flags = @($models[$model]) + @($configs[$name])
        if ($name -ne "full") { $flags += @("--crop", $Boxes) }
        python -m visual_search.run_baseline --dataset sop-val --data-dir $DataDir --device cuda `
            --out "$OutDir/${model}_$name.json" @flags | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "run failed for $model / $name" }
    }
}
