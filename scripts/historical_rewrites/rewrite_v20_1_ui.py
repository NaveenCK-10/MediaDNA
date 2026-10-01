import os

# 1. Update Analyze.tsx
path_analyze = 'c:/Users/navee/Desktop/Projects/MediaDna/OpenAVFF/frontend/src/pages/Analyze.tsx'
with open(path_analyze, 'r', encoding='utf-8') as f:
    analyze_content = f.read()

analyze_content = analyze_content.replace(
    'authenticity verification',
    'forensic analysis'
)

with open(path_analyze, 'w', encoding='utf-8') as f:
    f.write(analyze_content)


# 2. Update ExplainabilityDashboard.tsx
path_explain = 'c:/Users/navee/Desktop/Projects/MediaDna/OpenAVFF/frontend/src/components/ExplainabilityDashboard.tsx'
with open(path_explain, 'r', encoding='utf-8') as f:
    explain_content = f.read()

explain_content = explain_content.replace(
    'Occlusion Sensitivity Regions',
    'Model-Sensitive Regions'
)

explain_content = explain_content.replace(
    'OCCLUSION SENSITIVITY (TOP REGIONS)',
    'MODEL-SENSITIVE REGIONS (TOP)'
)

with open(path_explain, 'w', encoding='utf-8') as f:
    f.write(explain_content)

print("Updated UI components successfully.")
