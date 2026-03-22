param(
    [ValidateSet("epl", "laliga", "bundesliga", "ucl", "uel")]
    [string]$League = "epl",
    [ValidateSet("daily", "weekly")]
    [string]$Mode = "weekly",
    [int]$TopN = 20
)

$env:PYTHONPATH = "src"
$today = Get-Date
$lookaheadDays = if ($Mode -eq "daily") { 1 } else { 7 }
$dateFrom = $today.ToString("yyyy-MM-dd")
$dateTo = $today.AddDays($lookaheadDays).ToString("yyyy-MM-dd")
$competitionMap = @{
    "epl" = "EPL"
    "laliga" = "LaLiga"
    "bundesliga" = "Bundesliga"
    "ucl" = "UCL"
    "uel" = "UEL"
}
$competition = $competitionMap[$League]
$primary = "config/presets/${League}_1x2_config.json"
$extras = @(
    "config/presets/${League}_over05_config.json",
    "config/presets/${League}_over15_config.json",
    "config/presets/${League}_over25_config.json",
    "config/presets/${League}_btts_config.json",
    "config/presets/${League}_homewin_config.json"
)
python -m valuebetting fetch-epl-data --config $primary --date-from $dateFrom --date-to $dateTo --competitions $competition
python -m valuebetting score-upcoming --config $primary
python -m valuebetting report-value --config $primary --extra-configs $extras --top-n $TopN
