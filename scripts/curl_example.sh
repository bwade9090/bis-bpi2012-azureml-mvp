#!/usr/bin/env bash
set -euo pipefail

SCORING_URI="https://bpi2012-risk-endpoint.koreacentral.inference.ml.azure.com/score"
KEY=<PRIMARY_KEY>

cat > sample.json << 'JSON'
[
  {
    "case_length": 12,
    "n_resources": 4,
    "mean_inter_event_minutes": 35.2,
    "max_inter_event_minutes": 120.0,
    "n_weekend_events": 1,
    "first_event_hour": 10,
    "last_event_hour": 16
  }
]
JSON

curl -s -X POST "$SCORING_URI" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d @sample.json | jq .