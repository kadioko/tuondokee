param(
    [string]$Config = "config/example_config.json",
    [string[]]$ExtraConfigs = @(
        "config/over05_config.json",
        "config/over15_config.json",
        "config/over25_config.json",
        "config/btts_config.json"
    ),
    [int]$TopN = 20,
    [int]$LookaheadDays = 7,
    [string[]]$Competitions = @("EPL", "LaLiga", "Bundesliga", "UCL", "UEL")
)

$env:PYTHONPATH = "src"
$today = Get-Date
$dateFrom = $today.ToString("yyyy-MM-dd")
$dateTo = $today.AddDays($LookaheadDays).ToString("yyyy-MM-dd")
python -m valuebetting fetch-epl-data --config $Config --date-from $dateFrom --date-to $dateTo --competitions $Competitions
python -m valuebetting score-upcoming --config $Config
python -m valuebetting report-value --config $Config --extra-configs $ExtraConfigs --top-n $TopN
