
# Dubai Weather Safety Guidelines

**Policy name:** Dubai Weather Safety Guidelines  
**Policy version:** 2.0  
**Geographic scope:** Dubai, United Arab Emirates  
**Purpose:** Evidence-based weather risk assessment, public safety guidance and controlled alert recommendations.

---

## 1. Purpose and Governing Principles

These guidelines define how the weather monitoring agent assesses weather-related risks affecting residents, commuters, outdoor workers and other people in Dubai.

The agent must produce evidence-based, hazard-specific assessments rather than generic weather advice.

The following principles are mandatory:

1. Official instructions from the UAE National Center of Meteorology (NCM), NCEMA, Dubai Police, MoHRE and other competent authorities take priority over this document.
2. Weather measurements must be distinguished from forecasts, observations, warnings and assumptions.
3. A weather condition must not automatically be classified as dangerous simply because a measurement is elevated.
4. Every risk assessment must consider the nature of the hazard, affected population, exposure, duration and geographic impact.
5. Missing information must be reported as missing, not interpreted as evidence of safety.
6. The AI model must explain assessment results. It must not independently invent thresholds, change calculated risk scores or override official instructions.
7. A recommendation to issue an alert is not confirmation that an alert has been approved or sent.
8. The agent must not claim that these guidelines were applied unless the file was successfully retrieved and validated.

These guidelines are an internal operational framework. They are not an official UAE government warning standard.

## 2. Official Sources and Evidence Hierarchy

Use the following source hierarchy when evaluating weather and safety conditions.

### Tier 1: Official authorities

- UAE National Center of Meteorology (NCM): https://www.ncm.gov.ae/
- National Emergency Crisis and Disaster Management Authority (NCEMA): https://www.ncema.gov.ae/
- Dubai Police: https://www.dubaipolice.gov.ae/
- UAE Government Portal: https://u.ae/
- Ministry of Human Resources and Emiratisation (MoHRE): https://www.mohre.gov.ae/

Official warnings, emergency instructions and restrictions must be preserved accurately.

### Tier 2: Established supporting sources

Use reputable meteorological, scientific and public-health sources when official information does not provide sufficient detail.

Examples include established international meteorological organizations and recognized public-health institutions.

### Tier 3: Unverified information

Social media posts, unverified reports and isolated third-party claims may identify a potential issue but must not independently establish an official warning or confirmed hazard.

### Source requirements

For every important piece of evidence, record:

- Source name.
- Source category.
- Original publication or observation time, if available.
- Retrieval time.
- Geographic coverage.
- Measurement or warning details.
- Whether the information is current.
- Whether it is independently corroborated.

Do not describe third-party information as an official UAE warning.

## 3. Evidence Quality and Confidence

Classify evidence using the following levels.

| Level | Classification | Meaning |
|---|---|---|
| E1 | Official and verified | Current information from a competent authority |
| E2 | Reliable and corroborated | Reputable information supported by additional evidence |
| E3 | Unverified signal | A single source or unconfirmed report |
| E4 | Stale or conflicting | Outdated, inconsistent or questionable information |
| E0 | Missing | No usable evidence available |

Evidence quality and physical risk severity are separate concepts.

A potentially severe hazard supported by weak evidence must not be treated as harmless. It should be reported with appropriate uncertainty and escalated when warranted.

Do not calculate high confidence merely because multiple sources repeat the same original report.

Confidence must reflect:

- Source reliability.
- Data freshness.
- Geographic relevance.
- Measurement completeness.
- Independent corroboration.
- Consistency between sources.

If essential evidence is missing, mark the assessment incomplete.

## 4. Risk Assessment Dimensions

Assess each identified hazard using five dimensions.

Each dimension uses a scale from 1 to 5.

### 4.1 Likelihood (L)

Estimate how likely the hazard is to occur or continue during the assessment period.

| Score | Classification | Definition |
|---|---|---|
| 1 | Very unlikely | No meaningful indication of occurrence |
| 2 | Unlikely | Limited supporting evidence |
| 3 | Possible | Some evidence indicates potential occurrence |
| 4 | Likely | Strong evidence indicates occurrence |
| 5 | Almost certain | Hazard is already occurring or strongly expected |

### 4.2 Severity (S)

Estimate the potential consequences if the hazard affects people or infrastructure.

| Score | Classification | Definition |
|---|---|---|
| 1 | Negligible | Minimal inconvenience or disruption |
| 2 | Minor | Limited discomfort or minor disruption |
| 3 | Moderate | Meaningful safety concern or service disruption |
| 4 | Major | Potential for serious injury, significant disruption or damage |
| 5 | Critical | Potential for life-threatening consequences or widespread serious damage |

### 4.3 Exposure (E)

Estimate the number or vulnerability of people and assets potentially exposed.

| Score | Classification | Definition |
|---|---|---|
| 1 | Very limited | Few people or assets exposed |
| 2 | Limited | Small identifiable group exposed |
| 3 | Moderate | Several groups or locations exposed |
| 4 | High | Large population or important infrastructure exposed |
| 5 | Very high | Extensive population, critical infrastructure or highly vulnerable groups exposed |

Consider:

- Residents.
- Commuters.
- Outdoor workers.
- Children and older people.
- People with mobility limitations.
- Critical infrastructure.
- Road users and transport services.

Do not assume that every person in Dubai has the same exposure.

### 4.4 Duration (D)

Estimate how long the hazard is expected to persist.

| Score | Classification | Definition |
|---|---|---|
| 1 | Very short | Less than 1 hour |
| 2 | Short | 1 to less than 3 hours |
| 3 | Moderate | 3 to less than 6 hours |
| 4 | Long | 6 to less than 12 hours |
| 5 | Prolonged | 12 hours or longer |

Use the available forecast or observation period.

Do not estimate duration when the source does not support it.

### 4.5 Geographic Impact (G)

Estimate the geographic extent of the affected area.

| Score | Classification | Definition |
|---|---|---|
| 1 | Very localized | A single small location |
| 2 | Localized | A limited local area |
| 3 | Moderate | Multiple local areas |
| 4 | Extensive | Multiple districts or substantial parts of Dubai |
| 5 | Widespread | Widespread or emirate-level impact |

Do not infer district-level impact from a single city-wide weather reading.

## 5. Risk Calculation Model

The following is a provisional internal scoring model. It is not an official UAE government risk formula.

Calculate the score using:

Risk Score = 0.35 × (L × S) + 0.20 × (E × S) + 0.15 × (D × S) + 0.30 × (G × S)

Where:

- L = Likelihood
- S = Severity
- E = Exposure
- D = Duration
- G = Geographic impact

The resulting score ranges from 1 to 25.

### Risk bands

| Score | Risk band | Interpretation |
|---|---|---|
| 1 to 5 | LOW | Limited potential impact |
| Greater than 5 to 10 | MODERATE | Conditions require attention or monitoring |
| Greater than 10 to 15 | HIGH | Significant potential impact; review and precautionary action may be required |
| Greater than 15 to 25 | CRITICAL | Potentially severe impact requiring urgent assessment and escalation |

Use the unrounded score for classification. Round only the displayed value.

### Mandatory calculation rules

1. Every dimension must have an evidence-based score and a short rationale.
2. Do not assign scores based on intuition alone.
3. Do not treat missing evidence as a score of 0 or 1.
4. If an essential dimension cannot be assessed, return `UNASSESSED` rather than calculating a misleading complete score.
5. Record the evidence supporting each dimension.
6. Keep the numerical risk score separate from evidence confidence.
7. The same inputs and policy version must produce the same calculated result.
8. Official warnings and emergency instructions take precedence over the calculated risk band.

The scoring formula and bands must be validated against historical Dubai weather events and reviewed before being used for automated public notifications.

## 6. Weather Hazard Assessment

Assess each hazard independently. Multiple hazards may exist at the same time.

### 6.1 Extreme Heat

Evaluate:

- Maximum and current temperature.
- Feels-like or apparent temperature, where available.
- Relative humidity.
- Dew point, where available.
- Direct sunlight and solar exposure, where relevant.
- Wind conditions.
- Time of day.
- Expected duration.
- Outdoor activity and worker exposure.
- Relevant official heat advisories.

Temperature alone is insufficient to determine the full heat-related risk.

Higher humidity can reduce the body's ability to cool itself. Prolonged outdoor exposure can increase risk even when temperatures are below an extreme threshold.

Advice should be specific to the conditions and affected population.

Examples:

- Reduce strenuous outdoor activity during the hottest period.
- Take appropriate rest breaks.
- Use shaded or cooled areas.
- Maintain adequate hydration.
- Follow applicable workplace heat-safety requirements.
- Seek appropriate medical assistance when serious heat-illness symptoms occur.

### 6.2 Wind and Dust

Evaluate:

- Sustained wind speed.
- Wind gust speed.
- Wind direction.
- Expected duration.
- Dust or sandstorm observations.
- Visibility.
- Impact on roads, transport and outdoor activities.
- Relevant official warnings.

Distinguish sustained wind from short-duration gusts.

Do not apply an unsupported generic wind threshold as an official Dubai safety threshold.

If visibility is reduced, assess the potential impact on driving and outdoor activities.

Do not claim a road closure or transport disruption without authoritative supporting information.

### 6.3 Rain, Thunderstorms and Flooding

Evaluate:

- Expected rainfall intensity.
- Hourly rainfall, where available.
- Total rainfall.
- Expected duration.
- Thunderstorm warnings.
- Lightning risk.
- Drainage and geographic vulnerability, where known.
- Official flood or emergency instructions.
- Relevant road and transport information.

Daily rainfall totals alone do not establish the likelihood of localized flooding.

A city-wide weather forecast does not confirm that a particular road, tunnel or neighborhood is flooded.

Do not claim that flooding has occurred at a specific location without supporting evidence.

When an official warning is active, preserve its instructions and recommend appropriate precautions.

### 6.4 Low Temperatures and Cold Conditions

Evaluate:

- Minimum temperature.
- Feels-like temperature, where available.
- Wind conditions.
- Exposure duration.
- Vulnerable populations.
- Relevant official advisories.

Do not invent frost, freezing or cold-weather thresholds for Dubai.

Do not classify ordinary seasonal temperature variation as a significant safety hazard without supporting evidence.

### 6.5 Lightning and Thunderstorms

Evaluate:

- Official thunderstorm warnings.
- Lightning observations, where available.
- Rainfall intensity.
- Wind gusts.
- Outdoor exposure.
- Expected duration and geographic coverage.

If an official lightning or thunderstorm warning is issued, follow the authority's instructions.

Do not claim lightning is occurring at a specific location based only on a general forecast.

## 7. Regulatory and Workplace Compliance

Official regulatory requirements must be checked against the latest applicable government guidance.

For outdoor work in direct sunlight and open areas, consider the UAE midday break requirements.

The established midday break period generally applies from 12:30 PM to 3:00 PM between 15 June and 15 September, subject to the current applicable rules and exemptions.

The agent must:

1. Verify current requirements through MoHRE or another competent authority.
2. Distinguish workplace regulatory obligations from general public safety recommendations.
3. Avoid presenting workplace restrictions as a universal requirement for all residents.
4. Never invent exemptions or regulatory penalties.
5. Record the source and date of any compliance-related information.

Official regulatory instructions take priority over this document.

## 8. Alert Decision Framework

The agent must distinguish a risk assessment from an alert recommendation.

Use one of the following recommendation values:

- `NO_ALERT_RECOMMENDED`
- `MONITOR`
- `ADVISORY_RECOMMENDED`
- `ESCALATE_FOR_REVIEW`
- `OFFICIAL_WARNING_PASS_THROUGH`
- `UNABLE_TO_ASSESS`

### NO_ALERT_RECOMMENDED

Use only when:

- The required policy was successfully loaded.
- Relevant weather evidence is available.
- Essential risk dimensions are assessable.
- No relevant official warning or significant hazard requires escalation.
- The assessment is sufficiently complete.

This does not mean that weather conditions are guaranteed to be safe.

### MONITOR

Use when:

- A developing hazard is possible.
- Evidence indicates a changing condition.
- Further observation or verification is appropriate.
- No immediate advisory recommendation is justified by the available evidence.

### ADVISORY_RECOMMENDED

Use when:

- Evidence supports a meaningful safety concern.
- Practical precautionary advice is justified.
- The recommendation is proportionate to the hazard and exposure.

An advisory recommendation does not authorize external notification.

### ESCALATE_FOR_REVIEW

Use when:

- Potential consequences are serious.
- Evidence is conflicting or incomplete.
- A significant hazard may be developing.
- Human review or an authoritative source check is necessary.

### OFFICIAL_WARNING_PASS_THROUGH

Use when a current, relevant official warning or emergency instruction has been verified.

Preserve its original meaning, geographic coverage, validity period and issuing authority.

Do not rewrite official instructions in a way that changes their meaning.

### UNABLE_TO_ASSESS

Use when essential information is unavailable, invalid, stale or insufficient.

Examples include:

- Guidelines file unavailable.
- Weather API failure.
- Missing essential measurements.
- Unverified geographic coverage.
- Invalid or stale warning information.
- Incomplete evidence required for scoring.

Do not silently convert an incomplete assessment into a low-risk result.

## 9. Official Warning Override

Official warnings and emergency instructions must be treated as authoritative inputs.

If a competent authority issues a relevant warning:

1. Preserve the warning status.
2. Record the issuing authority.
3. Record the issue time and validity period when available.
4. Identify the geographic area covered.
5. Preserve the recommended safety actions.
6. Ensure the numerical risk model does not suppress the official warning.
7. Escalate where required by the configured notification and approval process.

An official warning may justify immediate precautionary communication even when the internal numerical risk score is incomplete or low.

The agent must never downgrade, dismiss or contradict an official emergency instruction.

## 10. Policy Retrieval and Validation

The agent must retrieve this document before performing a policy-based risk assessment.

Expected filename:

`guidelines.md`

The retrieval process must verify:

- The file exists.
- The file is readable.
- The file is not empty.
- The content can be parsed.
- The policy name and version are available.
- The content hash or equivalent version identifier is recorded.
- The retrieval time is recorded.

### If the guidelines file is unavailable

Set:

- `policy_status = UNAVAILABLE`
- `assessment_status = UNAVAILABLE`
- `alert_recommendation = UNABLE_TO_ASSESS`

Record the exact retrieval error.

The agent must not:

- Claim that these guidelines were applied.
- Substitute generic AI knowledge for the missing policy.
- Produce a definitive policy-based low-risk conclusion.
- Hide the retrieval failure.
- Claim a successful risk assessment.

A separately implemented and verified official-warning pass-through mechanism may still preserve an authoritative emergency instruction. It must not be represented as a completed policy assessment.

### If the guidelines file is available

Set:

- `policy_status = AVAILABLE`

Record the policy version, retrieval timestamp and content hash.

If the content is invalid or incomplete, report the validation failure rather than assuming that the policy is valid.

## 11. Data Quality and Assessment Completeness

Before producing an assessment, verify:

### Weather data

- City and geographic scope are correct.
- Coordinates or location configuration are valid.
- Units are identified.
- Measurements have timestamps.
- Forecast and observation values are distinguished.
- Data covers the assessment period.
- Unexpected or invalid values are rejected.
- Source retrieval errors are recorded.

### Risk data

- Each required dimension has a score or an explicit missing-data reason.
- Each score has a rationale.
- Evidence references are available.
- Conflicting information is disclosed.
- Risk calculation is reproducible.
- Confidence is not confused with severity.

### Assessment status

Use:

- `COMPLETE`
- `INCOMPLETE`
- `INVALID`
- `UNAVAILABLE`

`UNAVAILABLE` applies when the policy file cannot be retrieved (section 10); no assessment is attempted. Other failures keep their existing status until they are defined here.

A complete assessment requires the relevant policy, valid weather evidence and all essential dimensions for the hazard being evaluated.

An incomplete assessment must not be presented as a completed low-risk decision.

## 12. Required Structured Assessment Output

The agent should return a structured result with the following fields.

### Identification

- `assessment_id`
- `city`
- `geographic_scope`
- `assessment_timestamp`
- `timezone`
- `assessment_period`

### Policy and evidence

- `policy_status`
- `policy_version`
- `policy_hash`
- `policy_retrieval_error`
- `weather_source`
- `weather_retrieval_status`
- `weather_data_timestamp`
- `evidence_quality`
- `evidence_confidence`
- `evidence_references`
- `official_warning_status`
- `official_warning_authority`
- `official_warning_validity`

### Hazard assessment

- `hazard_type`
- `temperature_max_c`
- `temperature_min_c`
- `apparent_temperature_c`
- `relative_humidity_percent`
- `wind_speed_kmh`
- `wind_gust_kmh`
- `precipitation_mm`
- `visibility_km`

Include only measurements actually retrieved. Do not invent missing values.

### Risk dimensions

- `likelihood_score`
- `likelihood_reason`
- `severity_score`
- `severity_reason`
- `exposure_score`
- `exposure_reason`
- `duration_score`
- `duration_reason`
- `geographic_impact_score`
- `geographic_impact_reason`

### Decision

- `risk_score`
- `risk_band`
- `assessment_status`
- `alert_recommendation`
- `assessment_reasoning`
- `recommended_actions`
- `limitations`
- `missing_evidence`

### Notification control

- `approval_status`
- `notification_status`
- `notification_timestamp`
- `notification_error`

### Tool execution

- `tools_used`
- `tool_errors`

Use explicit null values or documented missing-data indicators where information is unavailable. Do not substitute invented values.

## 13. Advice and Communication Requirements

Advice must be:

- Specific to the identified hazard.
- Appropriate for the affected population.
- Proportionate to the assessed risk.
- Supported by available evidence.
- Consistent with official instructions.
- Clear about uncertainty and limitations.

Avoid generic advice that is unrelated to the identified hazard.

For example, a moderate wind assessment should explain the wind-related concern rather than automatically recommending hydration.

A heat assessment should explain the relevance of temperature, humidity, duration and exposure rather than relying only on a generic temperature statement.

When no significant hazard is identified, state what was assessed and why no alert is recommended.

Do not claim that all conditions are safe.

Do not claim that an alert has been sent unless the notification system confirms successful delivery.

## 14. Human Approval and Notification Governance

The agent may generate an assessment and recommend an alert.

External notification must follow the configured approval and authorization process.

Use these approval values:

- `PENDING_REVIEW`
- `APPROVED`
- `REJECTED`
- `NOT_AUTHORIZED`
- `NOT_REQUIRED`

Use these notification values:

- `NOT_SENT`
- `SENT`
- `FAILED`
- `NOT_APPLICABLE`

Rules:

1. Assessment generation does not equal approval.
2. Approval does not equal successful delivery.
3. Notification status must be based on the actual delivery result.
4. Failed notification attempts must be recorded.
5. Duplicate notifications must be controlled without suppressing a materially different hazard.
6. Official emergency instructions must follow the separately configured emergency communication procedure.
7. The AI model must not bypass approval or authorization controls.

## 15. Assessment Quality and Consistency Rules

Every assessment must pass the following checks:

- Correct city and geographic scope.
- Correct local timestamp and timezone.
- Valid and current weather evidence.
- Verified policy retrieval.
- Correct distinction between forecasts and observations.
- Hazard-specific analysis.
- Evidence-based scores.
- Reproducible risk calculation.
- Appropriate risk band.
- Official warning override respected.
- Missing information disclosed.
- Appropriate recommendation.
- No unsupported claims.
- No false notification confirmation.

The agent must not produce the same generic response for materially different weather conditions.

When conditions change, the assessment must explain which measurements, hazards, risk dimensions or recommendations changed.

When conditions remain similar, it may explain that the assessment is unchanged and identify the evidence supporting that conclusion.

## 16. Final Decision Rule

Before returning a result, answer these questions:

1. Was the correct guidelines file successfully loaded?
2. Is the weather data valid, current and relevant to Dubai?
3. Are the identified hazards supported by evidence?
4. Have all essential risk dimensions been evaluated?
5. Is the risk score reproducible?
6. Has evidence confidence been assessed separately?
7. Have official warnings and regulatory instructions been respected?
8. Is the recommendation proportionate to the evidence?
9. Are missing information and limitations disclosed?
10. Does the notification status reflect an actual system result?

If any essential requirement fails, return an incomplete assessment or escalate it.

Never fabricate evidence, risk scores, official warnings, approvals or notification results.

**End of policy.**
