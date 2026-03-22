param(
    [string]$PrimaryConfig = "config/example_config.json",
    [string[]]$ExtraConfigs = @(
        "config/over05_config.json",
        "config/over15_config.json",
        "config/over25_config.json",
        "config/btts_config.json"
    )
)

$env:PYTHONPATH = "src"
python -m valuebetting train-model --config $PrimaryConfig
foreach ($config in $ExtraConfigs) {
    python -m valuebetting train-model --config $config
}
