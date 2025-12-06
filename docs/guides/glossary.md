# GreenLang Process Heat Platform - Technical Glossary

**Document Version:** 1.0.0
**Last Updated:** 2025-12-06
**Classification:** Reference Documentation

---

## Table of Contents

1. [Process Heat Terminology](#1-process-heat-terminology)
2. [GreenLang-Specific Terms](#2-greenlang-specific-terms)
3. [Regulatory Acronyms](#3-regulatory-acronyms)
4. [ML/AI Terminology](#4-mlai-terminology)
5. [Industrial Communication](#5-industrial-communication)
6. [Safety and Compliance](#6-safety-and-compliance)

---

## 1. Process Heat Terminology

### Boiler and Steam Systems

| Term | Definition |
|------|------------|
| **Blowdown** | Intentional removal of water from a boiler to control dissolved solids concentration. Expressed as percentage of steam flow. Typical range: 1-5%. |
| **Boiler Efficiency** | Ratio of heat absorbed by water/steam to heat input from fuel. Expressed as percentage. Typical range: 80-90%. |
| **Carryover** | Transport of water droplets with steam leaving the boiler drum. Caused by high steam demand or improper water chemistry. |
| **Condensate** | Water formed when steam releases its latent heat. Recovered condensate reduces makeup water and energy requirements. |
| **Deaerator** | Equipment that removes dissolved oxygen and carbon dioxide from feedwater to prevent corrosion. Uses steam heating. |
| **Drum Level** | Water level in the steam drum. Critical safety parameter typically maintained +/- 2 inches of setpoint. |
| **Economizer** | Heat exchanger that uses flue gas heat to preheat feedwater. Improves efficiency by 2-5%. |
| **Feedwater** | Water supplied to the boiler, consisting of makeup water and returned condensate. |
| **Flash Steam** | Steam produced when high-pressure condensate is released to a lower pressure. |
| **Flue Gas** | Products of combustion exiting the boiler. Contains CO2, H2O, N2, excess O2, and pollutants. |
| **Header** | Large pipe that collects or distributes steam to/from multiple sources or users. |
| **Latent Heat** | Heat required to change water to steam at constant temperature. Approximately 970 BTU/lb at atmospheric pressure. |
| **Makeup Water** | Fresh water added to replace water lost through blowdown, leaks, and steam consumption. |
| **Saturated Steam** | Steam at the boiling point temperature for its pressure. Contains no superheat. |
| **Steam Trap** | Device that allows condensate to pass while preventing steam loss. Types: thermostatic, mechanical, thermodynamic. |
| **Superheat** | Additional heat added to steam above saturation temperature. Measured in degrees F above saturation. |
| **Surface Blowdown** | Continuous removal of water from the boiler drum surface to remove floating impurities. |

### Combustion

| Term | Definition |
|------|------------|
| **Air-Fuel Ratio** | Mass ratio of air to fuel in combustion. Stoichiometric for natural gas is approximately 17:1. |
| **Atomization** | Breaking liquid fuel into fine droplets for better mixing and combustion. |
| **Burner** | Device that mixes fuel and air and maintains stable combustion. Types: single-point, multi-point, low-NOx. |
| **CO (Carbon Monoxide)** | Product of incomplete combustion. Measured in ppm. Indicates combustion quality. Target: <100 ppm. |
| **Combustion Efficiency** | Percentage of fuel energy released as heat. Affected by excess air and combustion quality. |
| **Excess Air** | Air supplied beyond stoichiometric requirements. Expressed as percentage. Typical range: 10-20%. |
| **Flame Failure** | Loss of combustion flame. Triggers immediate fuel shutoff for safety. |
| **Flue Gas Temperature** | Temperature of exhaust gases. Higher temperature indicates more heat loss. |
| **Fuel-Bound Nitrogen** | Nitrogen contained in fuel that can form NOx during combustion. |
| **NOx (Nitrogen Oxides)** | Pollutant formed during high-temperature combustion. Regulated by EPA. |
| **O2 (Oxygen)** | Measured in flue gas to determine excess air. Typical target: 2-4% for natural gas. |
| **Stoichiometric** | Exact amount of air required for complete combustion with no excess. |
| **Thermal NOx** | NOx formed from atmospheric nitrogen at high temperatures (>2800 F). |
| **Turndown Ratio** | Ratio of maximum to minimum firing rate. Higher is better for flexibility. |

### Heat Transfer

| Term | Definition |
|------|------------|
| **Conduction** | Heat transfer through solid materials. Rate depends on conductivity and temperature difference. |
| **Convection** | Heat transfer by fluid motion. Dominant in economizers and air heaters. |
| **Fouling** | Buildup of deposits on heat transfer surfaces reducing effectiveness. |
| **Heat Exchanger** | Device for transferring heat between two fluids. Types: shell-tube, plate, finned tube. |
| **Heat Transfer Coefficient** | Rate of heat transfer per unit area per degree temperature difference. |
| **LMTD (Log Mean Temperature Difference)** | Average temperature difference used in heat exchanger calculations. |
| **NTU (Number of Transfer Units)** | Dimensionless measure of heat exchanger size and effectiveness. |
| **Radiation** | Heat transfer by electromagnetic waves. Dominant in furnace. |
| **Thermal Conductivity** | Material property indicating ability to conduct heat. Units: BTU/hr-ft-F. |
| **U-Value** | Overall heat transfer coefficient including all resistances. |

### Efficiency and Losses

| Term | Definition |
|------|------------|
| **Dry Flue Gas Loss** | Heat lost in sensible heat of dry flue gas. Largest loss, typically 4-8%. |
| **HHV (Higher Heating Value)** | Total heat released including condensation of water vapor. Used in US. |
| **LHV (Lower Heating Value)** | Heat released excluding water vapor condensation. Used in Europe. |
| **Moisture Loss** | Heat lost in vaporizing fuel moisture and combustion water. Typically 3-5%. |
| **Radiation Loss** | Heat lost from boiler surfaces. Typically 0.5-2%. Higher at low loads. |
| **Stack Loss** | Total heat lost in flue gas. Sum of dry gas and moisture losses. |
| **Unburned Loss** | Heat lost in unburned fuel (CO, combustibles). Should be <0.5%. |

---

## 2. GreenLang-Specific Terms

### Platform Components

| Term | Definition |
|------|------------|
| **Agent** | Autonomous software component that performs specific optimization or monitoring tasks. Each agent has a unique GL-XXX identifier. |
| **Agent Health Score** | Percentage (0-100%) indicating agent operational status based on heartbeat, errors, latency, and resource usage. |
| **Calculation Provenance** | Complete audit trail of a calculation including inputs, outputs, formulas, and SHA-256 hash for tamper detection. |
| **GreenLang Hub** | Central marketplace and community platform for GreenLang solutions, agents, and integrations. |
| **Orchestrator** | Central coordination agent (GL-001) that manages multi-agent workflows and system-wide operations. |
| **Process Heat Platform** | GreenLang's industrial thermal optimization system comprising multiple specialized agents. |
| **Provenance Hash** | SHA-256 cryptographic hash that uniquely identifies and validates a calculation for audit purposes. |
| **ThermalIQ** | GreenLang's calculation library providing validated engineering calculations with zero-hallucination guarantees. |
| **Workflow** | Sequence of coordinated agent tasks to accomplish a complex objective. |
| **Zero Hallucination** | Design principle ensuring calculations use only deterministic algorithms without AI/ML inference in critical paths. |

### Agent Identifiers

| ID | Name | Function |
|----|------|----------|
| **GL-001** | Thermal Command | Central orchestrator for multi-agent coordination |
| **GL-002** | Boiler Optimizer | Boiler efficiency optimization per ASME PTC 4.1 |
| **GL-003** | Steam Distribution | Steam header and distribution optimization |
| **GL-005** | Combustion Diagnostics | Combustion analysis and tuning recommendations |
| **GL-006** | Waste Heat Recovery | Heat recovery opportunity identification |
| **GL-007** | Furnace Monitor | Industrial furnace monitoring and optimization |
| **GL-008** | Steam Trap Monitor | Steam trap performance monitoring |
| **GL-009** | Thermal Fluid | Thermal oil system management |
| **GL-010** | Emissions Guardian | Real-time emissions monitoring per EPA Method 19 |
| **GL-011** | Fuel Optimization | Fuel blending and cost optimization |
| **GL-013** | Predictive Maintenance | Equipment failure prediction using Weibull analysis |
| **GL-014** | Heat Exchanger | Heat exchanger performance monitoring |
| **GL-015** | Insulation Analysis | Thermal insulation assessment |
| **GL-016** | Water Treatment | Boiler water chemistry management |
| **GL-017** | Condenser Optimization | Condenser performance optimization |
| **GL-018** | Combustion Control | Advanced burner control and O2 trim |

### Agent States

| State | Definition |
|-------|------------|
| **INITIALIZING** | Agent is starting up and running safety checks |
| **READY** | Agent has passed safety checks and is ready to process |
| **PROCESSING** | Agent is actively performing a calculation or task |
| **RUNNING** | Agent is operating normally in automatic mode |
| **WAITING** | Agent is waiting for input or external trigger |
| **ERROR** | Agent has encountered an error and requires attention |
| **SHUTDOWN** | Agent has been gracefully stopped |
| **EMERGENCY_STOP** | Agent has been stopped due to safety concern |
| **STANDBY** | Agent is available but not actively processing |

### Platform Features

| Term | Definition |
|------|------------|
| **Audit Logger** | System component that records all significant events for compliance and troubleshooting. |
| **Event Bus** | Message infrastructure for asynchronous communication between agents. |
| **Override** | Manual control input that supersedes automatic agent recommendations. |
| **Safety Guard** | Context manager that enforces pre/post operation safety checks. |
| **Watchdog Timer** | Timer that triggers alert if agent fails to send heartbeat within timeout period. |

---

## 3. Regulatory Acronyms

### Environmental Protection Agency (EPA)

| Acronym | Full Name | Definition |
|---------|-----------|------------|
| **CAA** | Clean Air Act | Federal law regulating air emissions in the United States. |
| **CAMD** | Clean Air Markets Division | EPA division managing emissions trading programs. |
| **CEMS** | Continuous Emission Monitoring System | Instruments that continuously measure pollutant emissions from sources. |
| **CFR** | Code of Federal Regulations | Codification of federal regulations including environmental rules. |
| **eGRID** | Emissions & Generation Resource Integrated Database | EPA database of environmental characteristics of electric power generation. |
| **GHG** | Greenhouse Gas | Gases that trap heat in atmosphere: CO2, CH4, N2O, HFCs, PFCs, SF6. |
| **GHGRP** | Greenhouse Gas Reporting Program | EPA program requiring large emitters to report GHG emissions annually. |
| **MACT** | Maximum Achievable Control Technology | Emission standards for hazardous air pollutants. |
| **NESHAP** | National Emission Standards for Hazardous Air Pollutants | EPA standards for HAP emissions under Clean Air Act. |
| **NSPS** | New Source Performance Standards | EPA emission limits for new and modified facilities. |
| **PM** | Particulate Matter | Solid particles or liquid droplets in air. PM2.5 and PM10 are regulated. |
| **PSD** | Prevention of Significant Deterioration | Program to prevent air quality degradation in clean areas. |
| **SIP** | State Implementation Plan | State plan for achieving national air quality standards. |
| **VOC** | Volatile Organic Compound | Carbon compounds that evaporate easily and contribute to smog. |

### Safety Standards

| Acronym | Full Name | Definition |
|---------|-----------|------------|
| **ANSI** | American National Standards Institute | Organization coordinating US voluntary standards. |
| **API** | American Petroleum Institute | Trade association developing standards for oil and gas industry. |
| **ASME** | American Society of Mechanical Engineers | Professional organization developing codes for pressure equipment. |
| **ESD** | Emergency Shutdown | System that rapidly shuts down equipment in emergency. |
| **IEC** | International Electrotechnical Commission | International standards organization for electrical and electronic technologies. |
| **ISA** | International Society of Automation | Professional organization for automation standards. |
| **NFPA** | National Fire Protection Association | Organization developing fire safety codes and standards. |
| **OSHA** | Occupational Safety and Health Administration | Federal agency ensuring workplace safety. |
| **PFD** | Probability of Failure on Demand | Safety metric indicating likelihood of safety system failure when needed. |
| **PSM** | Process Safety Management | OSHA regulation for managing hazardous chemical processes. |
| **RMP** | Risk Management Program | EPA program for facilities with hazardous substances. |
| **SIF** | Safety Instrumented Function | Automated action to achieve safe state. |
| **SIL** | Safety Integrity Level | Level of risk reduction provided by safety function (1-4). |
| **SIS** | Safety Instrumented System | System that performs safety functions automatically. |

### Industry Standards

| Acronym | Full Name | Application |
|---------|-----------|-------------|
| **ABMA** | American Boiler Manufacturers Association | Boiler industry standards and guidelines. |
| **ASME PTC 4** | Performance Test Code for Steam Generating Units | Standard method for boiler efficiency testing. |
| **API 560** | Fired Heaters for General Refinery Service | Standard for refinery fired heaters. |
| **IEEE** | Institute of Electrical and Electronics Engineers | Standards for electrical equipment. |
| **ISO 14001** | Environmental Management Systems | International environmental management standard. |
| **ISO 50001** | Energy Management Systems | International energy management standard. |
| **NIST** | National Institute of Standards and Technology | US measurement standards and technology agency. |

### Emissions Trading

| Acronym | Full Name | Definition |
|---------|-----------|------------|
| **CBAM** | Carbon Border Adjustment Mechanism | EU mechanism taxing carbon in imports. |
| **CORSIA** | Carbon Offsetting and Reduction Scheme for International Aviation | Aviation carbon offsetting program. |
| **ETS** | Emissions Trading System | Market-based system for trading emission allowances. |
| **EU ETS** | European Union Emissions Trading System | World's largest carbon market. |
| **GHG Protocol** | Greenhouse Gas Protocol | International standard for GHG accounting. |
| **RGGI** | Regional Greenhouse Gas Initiative | Northeast US cap-and-trade program. |
| **SBTi** | Science Based Targets initiative | Framework for setting corporate emission reduction targets. |
| **TCFD** | Task Force on Climate-related Financial Disclosures | Framework for climate risk disclosure. |

---

## 4. ML/AI Terminology

### Machine Learning Concepts

| Term | Definition |
|------|------------|
| **Anomaly Detection** | ML technique to identify unusual patterns that deviate from expected behavior. Used for equipment fault detection. |
| **Classification** | ML task of predicting which category an input belongs to. Example: predicting equipment failure mode. |
| **Clustering** | Unsupervised learning to group similar data points. Used for identifying operating regimes. |
| **Confidence Interval** | Range around prediction indicating uncertainty. 95% CI means 95% probability true value is within range. |
| **Data Drift** | Change in data distribution over time causing model performance degradation. |
| **Deep Learning** | ML using neural networks with multiple layers. Used for complex pattern recognition. |
| **Ensemble** | Combining multiple models for improved prediction accuracy. |
| **Feature Engineering** | Creating input variables from raw data to improve model performance. |
| **Feature Importance** | Measure of how much each input variable contributes to prediction. |
| **Gradient Boosting** | Ensemble technique building models sequentially to correct errors. |
| **Hyperparameter** | Model configuration set before training (learning rate, tree depth, etc.). |
| **Inference** | Using trained model to make predictions on new data. |
| **Model Registry** | System for storing, versioning, and managing ML models. |
| **Overfitting** | Model performs well on training data but poorly on new data. |
| **Random Forest** | Ensemble of decision trees using random subsets of features. |
| **Regression** | ML task of predicting continuous values. Example: predicting efficiency. |
| **Reinforcement Learning** | ML technique where agent learns by trial and error with rewards. |
| **Supervised Learning** | ML using labeled data where correct answers are known. |
| **Time Series** | Data points indexed by time. Common in process monitoring. |
| **Training** | Process of fitting model parameters to data. |
| **Transfer Learning** | Using knowledge from one task to improve learning on another task. |
| **Unsupervised Learning** | ML finding patterns in data without labeled examples. |
| **Validation** | Evaluating model on held-out data to check generalization. |

### Explainability and Trust

| Term | Definition |
|------|------------|
| **Explainability** | Ability to understand and explain how model makes predictions. Critical for regulatory acceptance. |
| **LIME** | Local Interpretable Model-agnostic Explanations. Technique for explaining individual predictions. |
| **SHAP** | SHapley Additive exPlanations. Method assigning importance values to features for each prediction. |
| **Black Box** | Model whose internal workings are not interpretable. Deep neural networks are often black boxes. |
| **White Box** | Model whose logic is transparent and interpretable. Linear regression, decision trees. |
| **Uncertainty Quantification** | Estimating confidence bounds around predictions. Important for safety-critical applications. |

### Predictive Maintenance Terms

| Term | Definition |
|------|------------|
| **Condition-Based Maintenance** | Maintenance triggered by equipment condition indicators rather than schedule. |
| **Failure Mode** | Specific way in which equipment can fail. Examples: bearing failure, seal leak. |
| **Mean Time Between Failures (MTBF)** | Average time between equipment failures. Reliability metric. |
| **Mean Time To Repair (MTTR)** | Average time to repair failed equipment. Maintainability metric. |
| **P-F Interval** | Time between detectable fault (P) and functional failure (F). Window for intervention. |
| **Remaining Useful Life (RUL)** | Predicted time until equipment failure. |
| **Reliability** | Probability equipment will perform required function for specified period. |
| **Weibull Analysis** | Statistical method for reliability analysis and failure prediction. |

---

## 5. Industrial Communication

### Protocols

| Term | Definition |
|------|------------|
| **DCS** | Distributed Control System. Industrial control system for process automation. |
| **DNP3** | Distributed Network Protocol. Communication protocol for utilities and SCADA. |
| **EtherNet/IP** | Industrial Ethernet protocol for manufacturing. Uses CIP (Common Industrial Protocol). |
| **Fieldbus** | Family of industrial protocols for device communication. Examples: Foundation Fieldbus, Profibus. |
| **gRPC** | Google Remote Procedure Call. High-performance communication protocol. |
| **HTTP/HTTPS** | Hypertext Transfer Protocol. Standard web communication protocol. |
| **IEC 61850** | Standard for substation automation communication. |
| **Kafka** | Distributed streaming platform for high-throughput data pipelines. |
| **Modbus** | Simple, open protocol for industrial devices. Variants: RTU (serial), TCP (Ethernet). |
| **MQTT** | Message Queuing Telemetry Transport. Lightweight publish/subscribe protocol for IoT. |
| **OPC-UA** | Open Platform Communications Unified Architecture. Modern industrial interoperability standard. |
| **REST API** | Representational State Transfer. Architectural style for web services. |
| **SCADA** | Supervisory Control and Data Acquisition. System for remote monitoring and control. |
| **WebSocket** | Protocol for full-duplex communication over single TCP connection. |

### Data Concepts

| Term | Definition |
|------|------------|
| **Data Lake** | Storage repository holding vast amounts of raw data in native format. |
| **Historian** | Database optimized for time-series data from industrial processes. |
| **JSON** | JavaScript Object Notation. Lightweight data interchange format. |
| **Namespace** | Hierarchical organization of tags in OPC-UA. |
| **Polling** | Periodically requesting data from a source. |
| **Quality Tag** | Indicator of data validity (Good, Bad, Uncertain, Stale). |
| **Subscription** | Receiving data automatically when values change. |
| **Tag** | Named reference to a data point in industrial systems. |
| **Timestamp** | Date and time associated with data value. |
| **XML** | Extensible Markup Language. Format for structured data. |
| **YAML** | YAML Ain't Markup Language. Human-readable configuration format. |

---

## 6. Safety and Compliance

### Safety Integrity Levels (SIL)

| Level | PFD Range | Risk Reduction | Typical Application |
|-------|-----------|----------------|---------------------|
| **SIL 1** | 10^-1 to 10^-2 | 10-100x | Minor injury prevention |
| **SIL 2** | 10^-2 to 10^-3 | 100-1,000x | Serious injury prevention |
| **SIL 3** | 10^-3 to 10^-4 | 1,000-10,000x | Fatality prevention |
| **SIL 4** | 10^-4 to 10^-5 | 10,000-100,000x | Catastrophe prevention |

### Safety System Terms

| Term | Definition |
|------|------------|
| **Alarm** | Indication of abnormal condition requiring operator attention. |
| **Alarm Flood** | Excessive number of alarms preventing effective response. |
| **Alarm Rationalization** | Process of reviewing and improving alarm settings. |
| **Annunciator** | Panel displaying alarm status with visual and audible indicators. |
| **Bypass** | Temporary disabling of safety function. Requires authorization and documentation. |
| **Demand** | Event requiring safety system action. |
| **Fail-Safe** | Design principle where failures result in safe state. |
| **First-Out** | Display showing which condition triggered sequence first. |
| **Hazard** | Source of potential harm. |
| **Interlock** | Automatic safety action preventing dangerous operation. |
| **LOPA** | Layer of Protection Analysis. Method for evaluating risk reduction. |
| **Override** | Operator action to supersede automatic control. |
| **Risk** | Combination of hazard probability and consequence severity. |
| **Safe State** | Condition where process is secure from identified hazards. |
| **Safeguard** | Measure reducing risk (barrier, alarm, procedure). |
| **Trip** | Automatic shutdown of equipment by safety system. |

### Compliance Terms

| Term | Definition |
|------|------------|
| **Audit Trail** | Chronological record of system activities for accountability. |
| **Calibration** | Adjusting instrument to give accurate readings against standard. |
| **Compliance** | Adherence to regulatory requirements and standards. |
| **Continuous Monitoring** | Ongoing measurement of parameters without interruption. |
| **Documentation** | Records providing evidence of compliance and procedures. |
| **Inspection** | Examination to verify condition and compliance. |
| **Management of Change (MOC)** | Process for evaluating and implementing changes safely. |
| **Non-Conformance** | Deviation from requirement or specification. |
| **Operating Procedure** | Documented steps for safe and effective operation. |
| **Permit** | Authorization to perform specific work or emit pollutants. |
| **Record Retention** | Requirements for how long records must be kept. |
| **Regulatory Agency** | Government body enforcing regulations (EPA, OSHA). |
| **Self-Certification** | Declaration of compliance without third-party verification. |
| **Third-Party Verification** | Independent confirmation of compliance. |
| **Variance** | Approved deviation from regulatory requirement. |

---

## Quick Reference Tables

### Unit Conversions

| From | To | Multiply By |
|------|-----|-------------|
| BTU | kJ | 1.055 |
| BTU/lb | kJ/kg | 2.326 |
| lb/hr | kg/hr | 0.4536 |
| psig | bar | 0.0689 |
| psig | kPa | 6.895 |
| degrees F | degrees C | (F - 32) / 1.8 |
| MMBTU | GJ | 1.055 |
| therms | MMBTU | 0.1 |

### Common Emission Factors (Natural Gas)

| Pollutant | Emission Factor | Units |
|-----------|-----------------|-------|
| CO2 | 53.06 | kg/MMBTU |
| CO2 | 117.0 | lb/MMBTU |
| NOx | 0.0092 | lb/MMBTU (uncontrolled) |
| NOx | 0.0032 | lb/MMBTU (low-NOx burner) |
| CO | 0.082 | lb/MMBTU |
| CH4 | 0.0022 | lb/MMBTU |

### Steam Properties Quick Reference

| Pressure (psig) | Sat. Temp (F) | Latent Heat (BTU/lb) |
|-----------------|---------------|---------------------|
| 0 | 212 | 970 |
| 15 | 250 | 945 |
| 50 | 298 | 912 |
| 100 | 338 | 881 |
| 150 | 366 | 857 |
| 200 | 388 | 837 |
| 250 | 406 | 820 |
| 300 | 422 | 802 |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-06 | GL-TechWriter | Initial release |

---

*For questions about terminology, contact support@greenlang.io*
