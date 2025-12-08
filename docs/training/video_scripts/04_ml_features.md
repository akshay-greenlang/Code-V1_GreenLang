# Video Tutorial 04: Machine Learning Features

## Video Metadata

- **Title:** GreenLang Machine Learning Features Deep Dive
- **Duration:** 30 minutes
- **Level:** Advanced
- **Audience:** Process Engineers, Data Scientists, Advanced Operators
- **Prerequisites:** Video 01-03, Basic ML concepts helpful

## Learning Objectives

By the end of this video, viewers will be able to:
1. Understand the ML models used in GreenLang
2. Interpret predictions and confidence intervals
3. Use anomaly detection effectively
4. Leverage explainability features
5. Configure and trigger model retraining

---

## Script

### SCENE 1: Opening (0:00 - 0:45)

**VISUAL:**
- GreenLang logo animation
- Title card: "Machine Learning Features"
- ML visualization graphics

**NARRATION:**
"Welcome back to the GreenLang tutorial series. In this video, we'll explore GreenLang's machine learning capabilities in depth.

Machine learning is at the heart of GreenLang's intelligence. It powers predictions, detects anomalies, explains decisions, and optimizes your process. Let's dive in."

**ACTION:**
- Logo animation
- Transition to ML visualization
- Display topic overview

---

### SCENE 2: ML Architecture Overview (0:45 - 3:00)

**VISUAL:**
- ML architecture diagram
- Model pipeline animation

**NARRATION:**
"Let's start with an overview of GreenLang's ML architecture.

The ML engine consists of several components working together.

First, the data pipeline. Raw sensor data is collected, cleaned, and transformed into features that ML models can use. This includes handling missing data, removing outliers, and creating derived features.

Next, the model layer. GreenLang includes multiple pre-trained models for different purposes: temperature prediction, anomaly detection, and optimization.

The inference engine runs models in real-time, generating predictions every few seconds as new data arrives.

The explainability layer provides human-understandable explanations for model outputs. This is crucial for building trust and enabling informed decisions.

And finally, the training pipeline handles model updates, either on a schedule or triggered by data drift."

**ACTION:**
- Display architecture diagram
- Animate data flow through each component
- Highlight each layer as it's described

---

### SCENE 3: Prediction Models (3:00 - 8:00)

**VISUAL:**
- Prediction dashboard
- Time-series forecast visualization

**NARRATION:**
"GreenLang's prediction models forecast future process conditions based on historical patterns.

Let's look at the prediction panel in our agent dashboard. Here you can see temperature predictions for the next hour.

The model shows the predicted value - 858 degrees - along with a confidence interval. The shaded area represents the range where the actual value is likely to fall with 95% confidence.

Notice how the confidence interval widens for predictions further in the future. This is normal - the further ahead we predict, the less certain we can be.

Let's examine the prediction details. Click on the prediction panel to expand it.

You can see predictions at different horizons: 15 minutes, 30 minutes, 1 hour, and beyond. Each has its own confidence score.

The model type shown here is LSTM, a deep learning model that's particularly good at capturing time-series patterns.

The 'Factors' section shows what's driving the prediction. In this case, the model expects temperature to rise slightly because fuel flow has increased. This transparency helps you understand and validate the prediction."

**ACTION:**
- Navigate to agent prediction panel
- Point to predicted value and confidence interval
- Click to expand details
- Show predictions at different horizons
- Highlight driving factors
- Show model information

---

### SCENE 4: Understanding Confidence Levels (8:00 - 11:00)

**VISUAL:**
- Confidence level examples
- Decision guidance chart

**NARRATION:**
"Understanding confidence levels is key to using predictions effectively.

GreenLang displays confidence as a percentage. Let's understand what different levels mean.

Above 90% confidence, the model is highly certain. For these predictions, you can act with confidence - for example, adjusting setpoints proactively or scheduling maintenance.

Between 70 and 90%, the model is moderately confident. Use these predictions for planning, but verify with other information before taking significant action.

Between 50 and 70%, confidence is low. The prediction provides useful context but shouldn't drive decisions on its own.

Below 50%, the model is uncertain. This often indicates unusual conditions the model hasn't seen before.

When confidence is low, GreenLang will show a warning. This is your signal to investigate - either the process is in an unusual state, or the model may need retraining.

The dashboard also shows confidence trends over time. Consistently falling confidence can indicate concept drift - the process is changing in ways the model wasn't trained for."

**ACTION:**
- Display confidence level chart
- Show examples of each level
- Demonstrate low-confidence warning
- Show confidence trend chart

---

### SCENE 5: Anomaly Detection (11:00 - 16:00)

**VISUAL:**
- Anomaly detection dashboard
- Real-time anomaly scores
- Alert examples

**NARRATION:**
"Anomaly detection is one of GreenLang's most powerful features. It identifies unusual patterns that might indicate problems - often before conventional alarms would trigger.

The anomaly detection panel shows a real-time score from 0 to 1. Values below 0.3 indicate normal operation. Between 0.3 and 0.7 is the attention zone. Above 0.7 triggers an anomaly alert.

GreenLang uses an Isolation Forest algorithm. Unlike threshold-based alarms, it learns what 'normal' looks like for your specific process and flags deviations from that norm.

Let's look at an example. Here, the temperature is within normal limits at 845 degrees. But the anomaly score is elevated at 0.65.

Clicking into the details, we can see why. The combination of temperature, fuel flow, and air flow is unusual. Individually, each parameter is fine, but together they form an unusual pattern.

This multivariate detection catches issues that single-parameter monitoring would miss. In this case, it might indicate a developing burner problem.

The anomaly history shows past detections. You can use this to identify recurring patterns or correlate with maintenance events.

For each anomaly, GreenLang provides a natural language explanation. Instead of just a score, you get a description: 'Temperature-fuel ratio deviates from normal pattern. Fuel flow is elevated relative to temperature increase. Possible causes: air damper restriction, burner efficiency degradation.'"

**ACTION:**
- Show anomaly panel with current score
- Explain the score scale
- Click into anomaly details
- Show contributing factors
- Display natural language explanation
- Review anomaly history

---

### SCENE 6: Explainability Features (16:00 - 21:00)

**VISUAL:**
- LIME explanation visualization
- Feature importance charts
- Natural language explanations

**NARRATION:**
"One of GreenLang's core principles is that AI should be explainable. Let's explore the explainability features.

GreenLang uses LIME - Local Interpretable Model-agnostic Explanations - to explain individual predictions. Let's see how this works.

Click on any prediction to see its explanation. The feature importance chart shows which inputs had the greatest influence on this specific prediction.

Here, we see that fuel flow rate contributed plus 12 degrees to the prediction. Inlet air temperature contributed plus 5 degrees. And product throughput contributed minus 3 degrees.

The visualization uses green bars for factors pushing the prediction up and red bars for factors pushing it down.

Below the chart is a natural language summary: 'Temperature is predicted to increase by 14 degrees over the next hour. This is primarily due to the 5% increase in fuel flow rate implemented at 14:30. The elevated inlet air temperature is also contributing. Consider reducing fuel input if temperature approaches the high limit.'

This isn't just data - it's actionable insight.

For anomaly detections, explanations work similarly. Instead of 'why this prediction,' you get 'why this is unusual.'

The attention visualization shows which time periods most influenced the model's decision. This helps you understand if the model is reacting to recent events or longer-term patterns."

**ACTION:**
- Click on prediction for explanation
- Show feature importance chart with bar colors
- Read natural language summary
- Navigate to anomaly explanation
- Display attention visualization

---

### SCENE 7: Causal Inference (21:00 - 24:00)

**VISUAL:**
- Causal graph visualization
- Root cause analysis example

**NARRATION:**
"Beyond correlation, GreenLang can help identify causal relationships using causal inference.

The causal analysis panel shows a graph of cause-and-effect relationships learned from your data.

In this example, we can see that fuel flow causally influences temperature, which makes physical sense. But we also see that product throughput influences both.

When investigating an issue, the causal analysis suggests root causes rather than just correlated factors.

Let's run a what-if analysis. I'll ask: what would happen if we reduced fuel flow by 5%?

The model predicts temperature would decrease by approximately 20 degrees, with a settling time of about 15 minutes.

This capability helps you plan changes before making them, reducing trial and error in your process optimization.

The causal graph evolves as more data is collected. GreenLang continuously refines its understanding of cause and effect in your specific process."

**ACTION:**
- Display causal graph
- Explain node relationships
- Run what-if analysis
- Show predicted outcome
- Demonstrate graph evolution concept

---

### SCENE 8: Model Management (24:00 - 27:00)

**VISUAL:**
- Model management dashboard
- Training configuration
- Performance metrics

**NARRATION:**
"GreenLang makes it easy to manage your ML models over time.

In the ML Management section, you can see all models deployed for your agents.

Each model shows its current version, last training date, and key performance metrics. The accuracy metric shows how well predictions match actual outcomes.

Data drift detection monitors whether incoming data is significantly different from training data. High drift suggests the model may need retraining.

To retrain a model, click the 'Retrain' button. You can retrain on-demand or set up automatic retraining.

For automatic retraining, set a schedule - weekly is common for most processes. You can also trigger retraining when drift exceeds a threshold.

When a new model is trained, GreenLang doesn't deploy it immediately. Instead, it runs challenger testing, comparing the new model's predictions against the current model.

If the new model performs better, it's promoted to production. If not, the current model continues. This ensures model updates improve rather than degrade performance."

**ACTION:**
- Navigate to ML Management
- Show model list with metrics
- Display drift indicators
- Click Retrain button
- Configure automatic retraining
- Explain champion-challenger concept

---

### SCENE 9: Optimization Recommendations (27:00 - 29:00)

**VISUAL:**
- Optimization panel
- Recommendation cards
- Implementation tracking

**NARRATION:**
"GreenLang's optimization module provides actionable recommendations to improve your process.

The optimization panel shows current recommendations ranked by potential impact.

Each recommendation includes the suggested change, expected benefit, and confidence level.

For example: 'Reduce Zone 1 fuel setpoint by 3%. Expected energy savings: 2.5%. Confidence: 85%. No impact on product quality predicted.'

You can apply recommendations with one click if automatic optimization is enabled, or use them as guidance for manual adjustments.

After implementing a change, GreenLang tracks the actual outcome and updates its models. This feedback loop continuously improves recommendation quality.

The optimization history shows past recommendations and their results. This transparency helps you build trust in the system's suggestions."

**ACTION:**
- Show optimization panel
- Review recommendation details
- Explain implementation options
- Show feedback tracking
- Display optimization history

---

### SCENE 10: Closing (29:00 - 30:00)

**VISUAL:**
- Summary and resources
- Next steps

**NARRATION:**
"In this video, we explored GreenLang's machine learning features in depth.

We covered prediction models and confidence interpretation, anomaly detection and multivariate analysis, explainability with LIME and natural language, causal inference for root cause analysis, model management and retraining, and optimization recommendations.

These capabilities transform GreenLang from a monitoring tool into an intelligent advisor for your process operations.

In our next video, we'll explore safety and compliance features - how GreenLang helps you meet regulatory requirements and maintain safe operations.

Thank you for watching. See you in the next tutorial!"

**ACTION:**
- Display topic summary
- Show next video card
- Fade to logo

---

## Production Notes

### Screen Recording Requirements

- ML dashboards with live data: 5+ minutes
- Explanation visualizations: Multiple examples
- Model management operations: Full walkthrough
- Optimization recommendations: Real examples

### Graphics Requirements

- ML architecture diagram (detailed)
- Confidence level guidance chart
- Causal graph animation
- Champion-challenger flow diagram

### Visualization Requirements

- Feature importance bar charts
- Time-series predictions with confidence bands
- Anomaly score timelines
- Attention heatmaps

### Technical Accuracy

- Have ML team review explanations
- Ensure examples are realistic
- Verify causal inference explanation is accurate

---

*Script Version: 1.0.0*
*Last Updated: December 2025*
