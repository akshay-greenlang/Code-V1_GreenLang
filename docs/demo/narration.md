# GreenLang CBAM Demo - Narration Script

This document provides the complete narration script for the GreenLang CBAM workflow demonstration video. Use this alongside the terminal recording for a polished presentation.

---

## Video Details

| Property | Value |
|----------|-------|
| **Title** | GreenLang CBAM Workflow Demo |
| **Duration** | ~4 minutes |
| **Target Audience** | Sustainability managers, compliance officers, developers |
| **Tone** | Professional, clear, confident |

---

## Scene 1: Introduction (0:00-0:15)

**[Terminal shows GreenLang ASCII logo]**

> "Welcome to GreenLang, the deterministic climate calculation platform."
>
> "In this demo, we'll walk through a complete CBAM workflow - from raw shipment data to a regulatory-ready declaration."
>
> "CBAM, the Carbon Border Adjustment Mechanism, is the EU's landmark policy for pricing carbon at the border. Let's see how GreenLang makes compliance straightforward."

---

## Scene 2: Environment Setup (0:15-0:35)

**[Terminal shows version and configuration]**

> "First, let's verify our GreenLang installation."
>
> "We can see we're running version 1.0 in production mode, with the audit trail enabled. This is critical for regulatory compliance - every calculation must be traceable."
>
> "GreenLang operates in deterministic mode by default, meaning the same inputs will always produce the exact same outputs. This is essential for third-party verification."

---

## Scene 3: Prepare Input Data (0:35-1:05)

**[Terminal displays sample shipment JSON]**

> "Here's our sample CBAM shipment record. This represents a real-world scenario: 500 tonnes of hot-rolled steel being imported from Turkey to Germany."
>
> "Notice the key data points: the CN code identifying the product, the production route - in this case, blast furnace with basic oxygen furnace - and the declared specific embedded emissions."
>
> "We also capture the exporter's installation ID for traceability, and transport details for a complete emissions picture."

---

## Scene 4: Validate Input Data (1:05-1:25)

**[Terminal shows validation output]**

> "Before running any calculations, GreenLang validates the input data against the official CBAM schema."
>
> "This catches errors early - invalid CN codes, malformed identifiers, or missing required fields. Our validation passes with no warnings, so we're ready to calculate."

---

## Scene 5: Run Emissions Calculation (1:25-1:55)

**[Terminal shows calculation progress and results]**

> "Now we execute the emissions calculation. Watch the pipeline: load data, resolve emission factors, calculate direct and indirect emissions, then add transport."
>
> "The result: 1,247 tonnes of CO2 equivalent embedded in this shipment. That breaks down to 925 tonnes direct emissions from the production process, 286 tonnes indirect from electricity use, and 37 tonnes from sea transport."
>
> "Under CBAM, the importer will need to purchase certificates covering these 1,247 tonnes."

---

## Scene 6: Examine Detailed Results (1:55-2:30)

**[Terminal displays detailed breakdown]**

> "Let's examine the detailed breakdown. The report shows exactly how each emission category was calculated."
>
> "Direct emissions include both combustion - burning fuels in the furnace - and process emissions from the chemical reactions in steelmaking."
>
> "Indirect emissions come from electricity consumption. Turkey's grid factor of 2.28 tonnes CO2 per megawatt-hour reflects their current energy mix."
>
> "Importantly, every emission factor is cited with its source. This transparency is what makes GreenLang audit-ready."

---

## Scene 7: Verify Provenance (2:30-3:05)

**[Terminal shows provenance record]**

> "This is where GreenLang really shines: full calculation provenance."
>
> "Every calculation gets a unique ID and timestamp. We record the exact algorithm versions, emission factor databases, and input file hashes."
>
> "The audit trail shows every step of the calculation pipeline. An auditor can trace exactly how we arrived at 1,247 tonnes - no black boxes."
>
> "Most importantly, the determinism check confirms this result is reproducible. Run it again with the same inputs, you'll get the same output - guaranteed."

---

## Scene 8: Verify Reproducibility (3:05-3:25)

**[Terminal shows verification output]**

> "Let's prove that reproducibility claim. The verify command recalculates from scratch and compares the results."
>
> "Original: 1,247.35 tonnes. Recalculated: 1,247.35 tonnes. Zero difference."
>
> "This verification passed. Any third-party auditor - regulators, certification bodies, or your own internal audit team - can independently verify this calculation."

---

## Scene 9: Export for Submission (3:25-3:45)

**[Terminal shows export completion]**

> "Finally, we export the declaration in the official EU CBAM XML format."
>
> "GreenLang generates a digitally signed submission package that's ready to upload directly to the EU CBAM Transitional Registry."
>
> "No manual data entry, no formatting errors - just a clean, compliant declaration."

---

## Scene 10: Summary (3:45-4:00)

**[Terminal shows workflow summary]**

> "That's the complete GreenLang CBAM workflow: validate, calculate, verify, and export."
>
> "Deterministic calculations. Full audit trails. Regulatory compliance built in."
>
> "Visit docs.greenlang.io to get started with your own CBAM compliance journey. Thanks for watching."

---

## Recording Notes

### Pacing

- Allow 2-3 seconds after each command before narration
- Pause at key numbers (emissions totals) for emphasis
- Speed up slightly during technical output scrolling

### Emphasis Points

- **Deterministic** - stress this word when it appears
- **1,247 tonnes** - pause for impact on the total
- **Zero difference** - emphasize in verification scene
- **Audit-ready** - confidence in voice

### Technical Notes

- If any command shows an error, re-record that scene
- Ensure terminal font is readable at 1080p
- Test playback at 1x and 1.5x speeds

### B-Roll Suggestions

If creating a polished video:

1. EU Parliament building (CBAM context)
2. Steel manufacturing facility
3. Cargo ship at port
4. Data center (representing calculations)
5. Auditor reviewing documents

---

## Accessibility

### Closed Captions

All narration should be transcribed for closed captions. Include:

- Speaker identification: "[NARRATOR]"
- Sound effects: "[Terminal typing sounds]"
- Visual descriptions: "[Green checkmark appears]"

### Audio Description

For visually impaired viewers, add audio descriptions:

- "The terminal displays a green ASCII art logo spelling GreenLang"
- "A JSON document appears showing shipment details"
- "Progress indicators show each calculation step completing"

---

## Translations

This script is the English master. Translations needed:

- [ ] German (DE) - Primary EU market
- [ ] French (FR) - EU market
- [ ] Spanish (ES) - EU market
- [ ] Mandarin (ZH) - For manufacturers
- [ ] Turkish (TR) - Key exporter market

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-09 | GL-TechWriter | Initial script |
