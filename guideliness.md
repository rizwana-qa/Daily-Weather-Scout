# Dubai Weather Safety, Risk Assessment and Alert Management Guidelines

**Location:** Dubai, United Arab Emirates  
**Policy purpose:** Weather monitoring, public safety risk assessment, alert generation, approval and notification management  
**Policy status:** Operational guidance for an AI-assisted weather monitoring system

## 1. Purpose and Safety Principles

These guidelines define how the weather monitoring agent must identify weather hazards, assess potential risks, generate alerts, obtain required approvals and communicate safety recommendations for Dubai, UAE.

The primary objective is to protect residents, commuters, outdoor workers and other potentially affected individuals.

Official instructions issued by the UAE National Center of Meteorology (NCM), NCEMA, Dubai Police, relevant UAE government authorities and emergency services always take priority.

The following principles are mandatory:

1. Human safety takes priority over notification frequency, convenience and automated processing.
2. Official warnings and instructions take priority over AI-generated assessments.
3. Every alert must be supported by identifiable evidence.
4. The agent must distinguish verified facts, forecasts, safety recommendations and assumptions.
5. Missing information must never be interpreted as confirmation that conditions are safe.
6. The agent must not invent warning thresholds, official announcements or safety instructions.
7. Alert severity and approval requirements must be determined before publication.
8. Every decision, approval, rejection, escalation and notification must be auditable.
9. When uncertainty exists, the agent must communicate that uncertainty clearly.
10. The system must fail safely rather than silently producing an unsupported no-alert decision.

## 2. Weather Hazard Categories

### Heat

Heatwave, scorching weather, scorching conditions, extreme heat and very high temperatures are related search terms for this section.

Issue a heat alert when NCM publishes a relevant heat warning or an official public-health advisory identifies dangerous heat conditions affecting Dubai.

During the UAE summer midday-break period, remind residents that outdoor work under direct sunlight and in open areas is prohibited from 12:30 PM to 3:00 PM, from 15 June to 15 September, subject to applicable official rules and exemptions.

This occupational requirement must not be misrepresented as a general requirement for all residents to remain indoors.

Recommend that residents:

- Limit strenuous outdoor activity during the hottest hours.
- Drink water regularly.
- Seek shaded or appropriately cooled environments.
- Check on children, older people and vulnerable individuals.
- Follow official heat-health advice.

**Safety reason:** Excessive heat can cause dehydration, heat exhaustion and potentially life-threatening heatstroke. People working or exercising outdoors may be particularly exposed.

Source: UAE Ministry of Human Resources and Emiratisation, Midday Break policy information:  
https://www.mohre.gov.ae/

### Wind

Strong winds, high winds, wind gusts and dust storms are related search terms for this section.

Issue a wind alert when NCM publishes a relevant warning affecting Dubai.

Use the following published UAE adverse-weather safety reference bands for additional risk classification:

- Wind gusts above 50 km/h: Strong wind conditions.
- Wind gusts above 75 km/h: Very strong wind conditions requiring an escalated risk assessment.

These are supplementary UAE safety reference bands, not official Dubai NCM warning thresholds. They must not override an official NCM warning or be presented as legally binding Dubai thresholds.

Advise residents to:

- Secure balcony furniture, umbrellas and loose outdoor objects.
- Avoid exposed outdoor locations.
- Stay away from unstable structures, trees, billboards and temporary installations.
- Reduce driving speed and maintain safe distances.
- Avoid unnecessary overtaking.
- Exercise additional caution when operating high-sided vehicles.

**Safety reason:** Strong winds can displace loose objects, damage temporary structures, reduce visibility and affect vehicle stability.

Source: Abu Dhabi Public Health Centre, ADOSH-SF Technical Guideline, Dealing with Adverse Weather Conditions, Version 4.0, 15 July 2024, Table Three, page 10:  
https://www.adphc.gov.ae/-/media/Project/ADPHC/ADPHC/PDF/OSHAD-SF/Technical-Guidelines/Dealing-with-Adverse-Weather-Eng.pdf

### Rain and Flooding

Heavy rain, flooding, flash floods, floodwater and water accumulation are related search terms for this section.

Issue a rain or flooding alert when NCM or another relevant UAE authority publishes a warning affecting Dubai, or an authoritative source confirms hazardous flooding or water accumulation.

Do not rely exclusively on daily rainfall totals. Short-duration heavy rainfall can create dangerous conditions even when daily rainfall totals are relatively low.

Recommend that residents:

- Avoid flooded roads, tunnels, underpasses and low-lying areas.
- Never drive or walk through moving or standing floodwater.
- Postpone unnecessary journeys when road conditions are unsafe.
- Keep mobile phones charged.
- Follow road closures and instructions from Dubai Police and emergency authorities.
- Move indoors during thunderstorms and avoid exposed outdoor locations.
- Avoid flooded locations where electrical hazards may exist.

After rainfall, warn residents about hidden hazards, damaged electrical equipment, debris and slippery surfaces.

**Safety reason:** Floodwater can conceal road damage, open drains and other obstacles. Moving water can sweep away people and vehicles. Flooding can also create electrical hazards.

Source: Abu Dhabi Public Health Centre, ADOSH-SF Technical Guideline, Dealing with Adverse Weather Conditions, Version 4.0, sections 4.4, 4.6 and 6.1:  
https://www.adphc.gov.ae/-/media/Project/ADPHC/ADPHC/PDF/OSHAD-SF/Technical-Guidelines/Dealing-with-Adverse-Weather-Eng.pdf

### Cold

Cold weather, low temperatures, frost and icy roads are related search terms for this section.

Issue a cold-weather alert when NCM publishes an official warning about unusually cold conditions, frost or potentially icy surfaces affecting Dubai.

Do not invent a routine minimum-temperature threshold without an authoritative Dubai-specific source.

Recommend:

- Warm clothing during unusually cold mornings and nights.
- Additional attention to infants, older people and individuals without adequate indoor heating.
- Reduced driving speed and additional caution when authorities warn about frost or icy roads.

Do not report frost or icy road conditions unless confirmed by an authoritative source.

**Safety reason:** Unusually cold conditions may affect vulnerable individuals. Frost and icy surfaces can increase the risk of slips and road accidents.

Source: UAE National Center of Meteorology:  
https://www.ncm.gov.ae

## 3. Source Authority and Validation

The agent must prioritize authoritative sources.

### Primary authority

- UAE National Center of Meteorology (NCM): https://www.ncm.gov.ae

### Additional authorities

- UAE Government: https://u.ae/en/information-and-services/environment-and-energy/weather-conditions-and-environmental-status
- NCEMA: https://www.ncema.gov.ae/
- Dubai Police: https://www.dubaipolice.gov.ae/
- UAE Ministry of Human Resources and Emiratisation: https://www.mohre.gov.ae/
- Abu Dhabi Public Health Centre: https://www.adphc.gov.ae/

The agent must validate:

1. Source identity and authority.
2. Publication or observation timestamp.
3. Applicable geographical coverage.
4. Forecast or warning validity period.
5. Hazard type and official warning classification.
6. Whether the warning is active, updated, cancelled or expired.
7. Whether the information is sufficiently current for the decision.

A source's authority must be evaluated according to the type of information being used. For example, MoHRE is relevant to employment regulations, while NCM is the primary source for official weather warnings.

Do not treat a third-party weather application, social media post or unverified report as an official authority announcement.

When sources conflict, prioritize the relevant competent authority and flag the discrepancy for review.

## 4. Risk Assessment and Alert Severity

The agent must classify the result of every assessment into one of four operational outcomes.

### Outcome A: Low Risk

Use only when:

- Required weather data is available and valid.
- Applicable guidelines have been successfully retrieved.
- Relevant official warnings have been checked.
- No applicable alert criteria or verified hazardous conditions have been identified.

The agent may recommend no alert.

### Outcome B: Advisory

Use when conditions require awareness and basic precautions but do not meet a higher warning classification.

Examples:

- An official advisory affecting Dubai.
- Weather conditions requiring additional precautions.
- A verified developing hazard with limited expected impact.

### Outcome C: Warning or Emergency Escalation

Use when an official warning or verified hazardous condition indicates a significant risk to people or property.

The agent must recommend protective action, identify affected locations and escalate according to the approval policy.

An emergency classification must not be assigned solely because an AI model considers a forecast severe. It must be supported by an authoritative emergency instruction or a verified situation meeting an explicitly approved emergency criterion.

### Outcome D: Assessment Incomplete

Use whenever critical evidence is unavailable, invalid or unreliable.

Examples:

- Safety guidelines cannot be retrieved.
- Weather API is unavailable.
- Weather data is stale or incomplete.
- The source cannot be verified.
- Sources contain unresolved conflicts.
- A relevant hazard section cannot be identified.
- Required location information is missing.

The agent must not convert an incomplete assessment into Low Risk.

It must record the reason, retry retrieval according to the configured retry policy and escalate unresolved failures to the responsible operator.

If available evidence indicates possible immediate danger, it must also raise a potential-hazard escalation rather than waiting for every dependency to recover.

## 5. Alert Decision Requirements

Before preparing an alert, the agent must establish:

- What weather hazard has been identified?
- Which geographical area is affected?
- What evidence supports the assessment?
- Is the evidence official or independently verified?
- When was the information published or observed?
- When is the hazard expected to occur?
- What is the expected or confirmed impact?
- Which safety guidelines apply?
- What protective actions should residents take?
- What severity is justified by the evidence?
- What approval is required before publication?

The agent must not generate an alert solely because a search keyword matches a guideline section.

A keyword match is retrieval evidence, not proof of a weather hazard.

## 6. Alert Approval and Authorization Policy

The AI agent must not have unrestricted authority to publish alerts.

The following approval workflow must be applied.

### Stage 1: Automated Assessment

The agent retrieves weather information, validates sources, retrieves applicable safety guidelines and evaluates the potential risk.

It prepares a structured assessment and recommends one of the operational outcomes.

No public notification is issued at this stage.

### Stage 2: Alert Draft

When an alert is justified, the agent prepares a draft containing:

- Alert title.
- Hazard category.
- Affected location.
- Proposed severity.
- Weather observations or forecast evidence.
- Official warning reference, if available.
- Expected timing and duration.
- Safety reason.
- Recommended precautions.
- Source links and timestamps.
- Assessment confidence and outstanding uncertainties.

The agent must distinguish official instructions from its own recommendations.

### Stage 3: Human Approval

Approval must be obtained from an authorized person before publication, according to the following internal governance matrix.

| Alert classification | Required authorization |
|---|---|
| Low Risk / No Alert | Automated assessment and logging |
| Advisory | Designated reviewer approval |
| Warning | Authorized safety or operations approver |
| Emergency | Designated emergency authority or on-call escalation process |
| Incomplete Assessment | No Low Risk approval; escalation or investigation required |

The organization must configure the actual approver roles and named authorized users before enabling publication.

The AI agent cannot approve its own alert.

### Stage 4: Publication

Only an approved alert may be published through the configured notification channels.

The system must record:

- Approver identity.
- Approval timestamp.
- Approved alert content and version.
- Publication timestamp.
- Notification destination or audience.
- Publication result.
- Any delivery failures.

Any material change to the severity, affected area or safety instructions requires renewed approval, unless an explicitly authorized emergency procedure applies.

### Stage 5: Emergency Escalation

If available evidence indicates imminent danger, the agent must immediately escalate to the designated emergency contact or authorized operator.

The system must not silently wait for a routine approval queue.

An organization may configure a narrowly defined, pre-authorized emergency notification procedure. This must have documented triggers, authorized recipients, approved message templates and audit logging.

The agent must never independently claim that an official government emergency alert has been issued.

## 7. Safety Reasons and Alert Content

Every alert must explain why the situation matters.

The safety reason must be connected to the actual hazard and expected impact.

Examples:

| Hazard | Safety reason |
|---|---|
| Extreme heat | Risk of dehydration, heat exhaustion and heatstroke |
| Strong winds | Risk from flying objects, unstable structures and vehicle instability |
| Dust storms | Reduced visibility and potential respiratory irritation |
| Heavy rain | Reduced road visibility, slippery surfaces and dangerous driving conditions |
| Flooding | Risk of drowning, vehicle entrapment, hidden obstacles and electrical hazards |
| Thunderstorms | Risk from lightning and exposure in open areas |
| Unusually cold conditions | Potential cold-related health effects for vulnerable individuals |
| Frost or icy surfaces | Increased risk of slips and road accidents |

The agent must not exaggerate the severity of a hazard or claim that an injury, accident or damage has occurred unless verified.

Safety recommendations must be practical, clear and proportionate.

## 8. Alert Message Format

Every proposed alert must follow a consistent format.

**Alert title:** Clear description of the hazard.

**Location:** Affected geographical area.

**Severity:** Advisory, Warning or Emergency Escalation.

**Assessment status:** Verified, Potential Hazard or Assessment Incomplete.

**Time:** Publication time, expected start and validity period where available.

**Weather evidence:** Verified measurements, forecast information or official warning.

**Official source:** Authority name and source reference.

**Safety reason:** Why the condition creates a potential risk.

**Recommended action:** Specific precautions residents should take.

**Approval status:** Pending, Approved, Rejected or Emergency Escalated.

**Limitations:** Missing evidence, uncertainties or conflicting information.

Never label an AI-generated assessment as an official NCM or government alert.

## 9. Alert Frequency, Updates and Closure

Generate alerts only when a qualifying official warning, advisory or verified local hazard meets the conditions defined in these guidelines.

Do not send repeated notifications about the same weather condition in the same affected area when severity and recommended actions have not changed.

Send an updated alert when:

- The official warning level increases.
- The affected geographical area expands.
- Expected timing or duration changes significantly.
- New safety instructions are issued.
- A previously reported condition becomes more dangerous.
- Previously unavailable evidence materially changes the assessment.

Never suppress an urgent safety escalation merely to prevent duplicate notifications.

Close or expire alerts when the official warning is cancelled or the relevant hazard period ends.

The agent must not automatically close an alert merely because an API stops reporting the hazard.

## 10. Failure Handling and Safe Operation

The following behaviours are mandatory.

| Failure condition | Required behaviour |
|---|---|
| Weather API unavailable | Mark assessment incomplete |
| Guidelines unavailable | Mark assessment incomplete |
| Guidelines outdated | Reject them for decision-making and escalate |
| No matching guideline section | Report no match; do not infer low risk |
| Invalid weather readings | Reject invalid data |
| Stale weather information | Mark information stale |
| Conflicting sources | Flag conflict and seek authoritative confirmation |
| Official warning unavailable | Do not invent an official warning |
| Notification delivery failure | Log failure and retry according to policy |
| Approval unavailable | Keep alert pending and escalate if necessary |
| Unexpected system exception | Capture error, preserve evidence and mark assessment incomplete |

A failed retrieval, API call, database operation or notification must never be silently treated as a successful safety assessment.

The agent must preserve the last known valid evidence separately from current evidence and must clearly identify its timestamp.

Retries must be bounded and logged. Repeated failures must not cause infinite loops or duplicate notifications.

## 11. Audit Trail and Compliance Controls

Every assessment must produce an auditable record.

The record must include:

- Unique assessment ID.
- Location.
- Assessment timestamp.
- Weather data source and observation time.
- Retrieved guideline version.
- Hazard category.
- Source authority and source reference.
- Risk classification.
- Safety reason.
- Alert recommendation.
- Evidence and uncertainties.
- Approval decision and approver identity.
- Publication or escalation result.
- Retry and failure history.
- Alert closure reason and timestamp.

Records must be protected against unauthorized modification.

The organization must define access permissions, retention periods and review responsibilities according to its applicable policies and legal requirements.

The agent must not expose personal information or sensitive operational details in public alerts.

## 12. Search and Retrieval Guidance

The section headings are intentional retrieval boundaries. Keep each weather hazard's rules under its own heading.

Alternative words and phrases help keyword retrieval but are not independent alert triggers.

A keyword match does not prove that a weather condition exists or that an alert threshold has been crossed.

The agent must retrieve the correct hazard guidance, preserve its source attribution and separately validate the weather evidence.

Keep these sections separate:

- Heat-related guidance under `## Heat`.
- Wind-related guidance under `## Wind`.
- Rain and flooding guidance under `## Rain and Flooding`.
- Cold-related guidance under `## Cold`.

Risk assessment, approval, failure-handling and audit requirements must remain separate from individual hazard guidance.

Do not delete or merge the headings during document preparation.

If no section matches a query, return a clear no-match result. Do not invent guidance or interpret missing search results as proof that no weather hazard exists.

## 13. Policy Governance and Review

The organization must maintain a controlled version of these guidelines.

Each policy version must have:

- Version number.
- Effective date.
- Review date.
- Document owner.
- Approval authority.
- Change history.

Review the guidelines whenever relevant UAE authorities update their warnings, regulations or safety guidance.

Numerical thresholds must have traceable sources and documented applicability.

Do not automatically adopt thresholds from another emirate or jurisdiction as Dubai-specific official thresholds.

Changes to alert severity, approval rules, emergency escalation or publication permissions require authorized review before deployment.

## 14. Final Safety Rule

Protecting people takes priority over automated reporting and system availability.

The agent must never:

- Treat missing guidelines as evidence of safety.
- Treat unavailable weather data as normal weather.
- Invent official warnings or measurements.
- Publish an unapproved alert outside an explicitly authorized emergency procedure.
- Suppress a genuine escalation because of duplicate prevention.
- Present an uncertain assessment as a verified fact.
- Override official instructions with AI-generated recommendations.

When reliable evidence is insufficient, the agent must explicitly report:

**ASSESSMENT INCOMPLETE: Weather safety could not be fully verified. No definitive low-risk conclusion is permitted. Refer to current official UAE authority updates and escalate for review where necessary.**

The system must prioritize verified evidence, safe decisions, authorized approvals, transparent communication and complete auditability.
