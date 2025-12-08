# Video Tutorial 03: Creating Your First Agent

## Video Metadata

- **Title:** Creating Your First Process Heat Agent
- **Duration:** 25 minutes
- **Level:** Intermediate
- **Audience:** Operators, Process Engineers, Administrators
- **Prerequisites:** Video 01 (Introduction), Video 02 (Installation)

## Learning Objectives

By the end of this video, viewers will be able to:
1. Create a new process heat monitoring agent
2. Configure agent parameters and thresholds
3. Set up basic alarms
4. Monitor agent output in real-time
5. Interpret agent data and alerts

---

## Script

### SCENE 1: Opening (0:00 - 0:30)

**VISUAL:**
- GreenLang logo animation
- Title card: "Creating Your First Agent"

**NARRATION:**
"Welcome back to the GreenLang tutorial series. In this video, we'll create our first process heat monitoring agent. By the end, you'll have a fully functional agent collecting data and providing intelligent insights."

**ACTION:**
- Logo animation
- Transition to title card

---

### SCENE 2: What is an Agent? (0:30 - 2:00)

**VISUAL:**
- Agent architecture diagram
- Animation showing agent components

**NARRATION:**
"Before we create an agent, let's understand what it does.

A GreenLang agent is an intelligent software module that monitors a specific part of your process. For process heat applications, an agent might monitor a single furnace zone, a heat treatment line, or an entire thermal system.

Each agent has four key capabilities.

First, data collection. The agent gathers data from your sensors and control systems in real-time.

Second, analysis. It processes this data, applying machine learning models to detect patterns and anomalies.

Third, prediction. Using historical patterns, the agent forecasts future conditions.

And fourth, alerting. When conditions warrant attention, the agent generates appropriate alarms."

**ACTION:**
- Display agent architecture diagram
- Animate each capability as it's described
- Show data flow through agent

---

### SCENE 3: Choosing Agent Type (2:00 - 4:00)

**VISUAL:**
- Dashboard agent templates screen
- Template comparison table

**NARRATION:**
"GreenLang offers several agent templates to match different process heat applications.

The Process Heat Monitoring template is the most common. It monitors temperature, fuel flow, and other parameters for a single zone or unit.

The Multi-Zone Coordinator template manages multiple zones together, optimizing their combined operation.

The Heat Treatment Agent is specialized for batch processes like annealing, tempering, or heat treatment cycles.

And the Combustion Optimizer focuses specifically on burner efficiency and emissions.

For this tutorial, we'll use the Process Heat Monitoring template. It's versatile and provides a great foundation for understanding GreenLang."

**ACTION:**
- Navigate to Agents > Templates
- Display template cards
- Highlight Process Heat Monitoring template
- Show template details

---

### SCENE 4: Creating the Agent - Basic Settings (4:00 - 7:00)

**VISUAL:**
- Agent creation wizard
- Form fields being filled

**NARRATION:**
"Let's create our agent. In the dashboard, click 'Agents' in the navigation, then click the plus button or 'New Agent'.

Select the 'Process Heat Monitoring' template and click Continue.

Now we'll configure the basic settings.

For the agent name, use something descriptive that identifies the specific equipment. I'll use 'furnace_zone_1' for this example.

Add a description to help others understand what this agent monitors. Something like 'Primary heating zone for Furnace A'.

The location field is optional but helpful for larger facilities. I'll enter 'Building 2, Bay 3'.

For the agent group, you can organize related agents together. I'll put this in the 'Furnace A' group.

These labels help organize your agents as your deployment grows. Click Continue to proceed."

**ACTION:**
- Click Agents in navigation
- Click New Agent button
- Select Process Heat Monitoring template
- Fill in name: "furnace_zone_1"
- Fill in description: "Primary heating zone for Furnace A"
- Fill in location: "Building 2, Bay 3"
- Select group: "Furnace A"
- Click Continue

---

### SCENE 5: Data Source Configuration (7:00 - 11:00)

**VISUAL:**
- Data source configuration panel
- Connection testing

**NARRATION:**
"Next, we configure data sources. This tells the agent where to get its data.

For this tutorial, we'll use the built-in simulator, which generates realistic process data. In a real deployment, you'd connect to your OPC-UA server, Modbus devices, or other data sources.

Click 'Add Data Source'. Select 'Simulator' as the type.

For the temperature source, we'll set a base temperature of 850 degrees Celsius with normal variation of plus or minus 15 degrees. This simulates typical furnace zone behavior.

Let's add a second data source for fuel flow. Click 'Add Data Source' again. Set the type to 'Simulator' and configure it to represent fuel flow as a percentage, with a base of 85% and variation of plus or minus 10%.

You can add more data sources as needed. Common ones include air flow, pressure, and product temperature.

GreenLang also supports calculated values. You can create derived metrics like heat input or efficiency from raw sensor data.

Click 'Test Connections' to verify the data sources are working. For simulators, this always succeeds, but for real connections, this confirms communication with your equipment."

**ACTION:**
- Click Add Data Source
- Select type: Simulator
- Configure temperature: Base 850, Variation 15
- Click Add Data Source again
- Configure fuel flow: Base 85%, Variation 10%
- Click Test Connections
- Show success message

---

### SCENE 6: Process Parameters (11:00 - 14:00)

**VISUAL:**
- Process parameters configuration
- Temperature setpoint and limits

**NARRATION:**
"Now we configure the process parameters. These define normal operating conditions and acceptable limits.

For temperature, set the setpoint to 850 degrees - this is your target temperature.

Set the high-high limit to 920 degrees. This is a critical safety limit that should trigger immediate action.

Set the high limit to 900 degrees. This is a warning that things are getting hot.

Set the low limit to 800 degrees. Below this, product quality may be affected.

Set the low-low limit to 750 degrees. This indicates a serious underheating condition.

You can configure similar parameters for fuel flow and other variables.

The rate of change limits are also important. I'll set these to plus or minus 5 degrees per minute. If temperature changes faster than this, it may indicate a problem.

These parameters are used for alarm generation and also feed into the machine learning models."

**ACTION:**
- Set temperature setpoint: 850
- Set high-high limit: 920
- Set high limit: 900
- Set low limit: 800
- Set low-low limit: 750
- Set rate of change limits: +/- 5 degrees/min
- Configure fuel flow parameters similarly

---

### SCENE 7: Alarm Configuration (14:00 - 17:00)

**VISUAL:**
- Alarm configuration panel
- Priority settings
- Notification channels

**NARRATION:**
"GreenLang automatically creates alarms based on the limits you set, but let's review and customize them.

The high-high temperature alarm is set to Critical priority. This means it will appear in red on the dashboard and send immediate notifications.

The high temperature alarm is set to High priority - orange on the dashboard, and it requires response within 5 minutes.

You can customize the alarm message. Click on the high temperature alarm. Change the message to something specific, like 'Zone 1 temperature exceeding target. Check fuel input and cooling system.'

For notifications, select which channels should receive alerts. For critical alarms, I recommend enabling all channels - dashboard, email, and SMS if configured.

You can also set up alarm delays. For example, if you don't want an alarm until the condition persists for 30 seconds, set the delay here. This helps avoid nuisance alarms from brief spikes.

The alarm suppression settings let you prevent alarm floods. If one alarm is active, you can suppress related alarms for a configured period."

**ACTION:**
- Review auto-generated alarms
- Click on High Temperature alarm
- Edit message text
- Configure notification channels
- Set delay: 30 seconds
- Show suppression settings
- Click Continue

---

### SCENE 8: ML Configuration (17:00 - 20:00)

**VISUAL:**
- Machine learning settings
- Model selection
- Prediction visualization

**NARRATION:**
"Now we configure the machine learning features.

GreenLang includes several pre-trained models optimized for process heat applications.

For temperature prediction, select the LSTM Temperature Predictor. This model uses historical patterns to forecast future temperatures.

Set the prediction horizon to 1 hour. The agent will continuously predict temperatures up to one hour ahead.

For anomaly detection, select Isolation Forest. This unsupervised model detects unusual patterns without needing labeled training data.

Set the sensitivity to Medium. Higher sensitivity catches more anomalies but may generate more false alerts.

Enable auto-retraining to let the model adapt to your specific process over time. The model will retrain weekly using your actual data.

The optimization module suggests setpoint adjustments to improve efficiency. Enable this if you want proactive recommendations.

These settings can be fine-tuned later as you learn how the agent performs with your specific process."

**ACTION:**
- Select temperature prediction model: LSTM Temperature Predictor
- Set horizon: 1 hour
- Select anomaly model: Isolation Forest
- Set sensitivity: Medium
- Enable auto-retraining: Weekly
- Enable optimization suggestions
- Click Continue

---

### SCENE 9: Review and Create (20:00 - 21:30)

**VISUAL:**
- Configuration summary screen
- Agent creation confirmation

**NARRATION:**
"Before creating the agent, let's review our configuration.

The summary screen shows all the settings we've configured. Review each section to make sure everything looks correct.

Agent name: furnace_zone_1
Data sources: Temperature and fuel flow simulators
Temperature setpoint: 850 degrees with appropriate limits
Alarms: Configured with custom messages and notifications
ML: LSTM prediction with Isolation Forest anomaly detection

If everything looks good, click 'Create Agent'.

GreenLang will initialize the agent, connect to data sources, and begin collecting data. This usually takes just a few seconds."

**ACTION:**
- Review summary screen
- Verify each configuration section
- Click Create Agent
- Show success message
- Watch progress indicator

---

### SCENE 10: Agent in Action (21:30 - 24:00)

**VISUAL:**
- Live agent dashboard
- Real-time data updating
- Predictions and alerts

**NARRATION:**
"Our agent is now live and running. Let's see it in action.

The agent dashboard shows real-time data from our simulated furnace zone. The temperature is updating every second, and you can see the trend chart building.

The current reading shows 847 degrees, just slightly below setpoint. The status indicator is green, indicating normal operation.

Look at the predictions panel on the right. The ML model is already making forecasts. It predicts the temperature will remain stable around 850 degrees for the next hour, with high confidence.

The anomaly score is low at 0.12, indicating the process is behaving normally.

Let's generate a test alarm. I'll use the testing feature to simulate a high temperature condition.

Watch the dashboard - the temperature spikes to 905 degrees. The high temperature alarm activates, turning the zone indicator orange. You'll also see the notification appear.

The agent immediately begins analyzing the situation. In a real scenario, it would provide root cause suggestions based on ML analysis.

I'll clear the test condition, and the alarm returns to normal."

**ACTION:**
- Navigate to agent dashboard
- Point out real-time readings
- Highlight trend chart
- Show prediction panel
- Show anomaly score
- Trigger test alarm
- Watch alarm activate
- Clear test condition

---

### SCENE 11: Closing (24:00 - 25:00)

**VISUAL:**
- Summary and next steps
- Resources

**NARRATION:**
"Congratulations! You've created your first GreenLang process heat agent.

We covered agent types and templates, data source configuration, process parameters and limits, alarm setup and customization, and machine learning configuration.

Your agent is now monitoring your process, making predictions, and ready to alert you to issues.

In the next video, we'll dive deeper into GreenLang's machine learning features, including how to interpret predictions, use explainability, and optimize your process.

Thank you for watching. See you in the next tutorial!"

**ACTION:**
- Display summary checklist
- Show next video card
- Fade to logo

---

## Production Notes

### Screen Recording Requirements

- Agent creation wizard: Full walkthrough
- Live dashboard: At least 2 minutes of real-time data
- Test alarm demonstration: Clear before and after

### Graphics Requirements

- Agent architecture diagram (animated)
- Template comparison table
- Configuration flow diagram

### Interactive Elements

- Highlight mouse clicks
- Zoom on small form fields
- Use callouts for important settings

### Timing Considerations

- Pause on form fields for readability
- Allow dashboard to populate with data before commenting
- Clear transitions between configuration steps

---

*Script Version: 1.0.0*
*Last Updated: December 2025*
