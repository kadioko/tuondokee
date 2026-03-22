param(
    [ValidateSet("epl", "laliga", "bundesliga", "ucl", "uel")]
    [string]$League = "epl"
)

$env:PYTHONPATH = "src"
$configs = @(
    "config/presets/${League}_1x2_config.json",
    "config/presets/${League}_over05_config.json",
    "config/presets/${League}_over15_config.json",
    "config/presets/${League}_over25_config.json",
    "config/presets/${League}_btts_config.json",
    "config/presets/${League}_homewin_config.json"
)
foreach ($config in $configs) {
    python -m valuebetting train-model --config $config
}
