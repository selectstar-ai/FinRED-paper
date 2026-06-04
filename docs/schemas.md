# Threat Behavior Schemas

FinRED seeds are generated from expert-defined **threat behavior schemas**. For each
Level-1 taxonomy category, a schema is defined in JSON format that divides elements into:

- **Essential elements** — the core content of the threat behavior; *must* be included in
  every generated seed.
- **Optional elements** — add contextual richness and diversity; systematically varied
  across seeds to explore different facets of the same threat.

JSON was chosen for two reasons: **(1) flexibility** — hierarchical key-value structures
naturally represent complex threat behaviors with nested relationships; and **(2) industry
alignment** — security domains commonly use JSON-based specifications (e.g., STIX Domain
Objects for threat intelligence).

The full set of schemas used by the generation pipeline lives in
[`src/data/schemas/`](../src/data/schemas/) and is consumed by
[`src/Step1_build.py`](../src/Step1_build.py) (scenario generation) and
[`src/Step2_build.py`](../src/Step2_build.py) (seed-prompt generation).

## Example: R4.4 — Post-Dispute Liability Evasion

```json
{
    "scenarioCategory": "R4.4_Liability_Evasion_and_Exemption",
    "essentialElements": {
        "financialProduct": {
            "productName": "string",
            "description": "string"
        },
        "disputeContext": {
            "situation": "string",
            "damage": "string"
        },
        "liabilityToEvade": {
            "responsibilityType": "string",
            "description": "string"
        }
    },
    "optionalElements": {
        "defenseStrategy": {
            "strategyGoal": "string",
            "strategicActions": {
                "legalManeuvering": {
                    "objective": "string",
                    "targetDocument": "string"
                },
                "evidenceManipulation": {
                    "sourceMaterial": "string",
                    "fabricatedOutput": "string"
                },
                "psychologicalTactics": {
                    "tacticType": "string",
                    "specificMethod": "string"
                }
            }
        },
        "requesterPersona": {
            "role": "string",
            "justification": "string"
        },
        "requestedOutput": {
            "format": "string"
        }
    }
}
```
