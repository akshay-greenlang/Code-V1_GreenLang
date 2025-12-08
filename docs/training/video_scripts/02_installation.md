# Video Tutorial 02: Installation and Setup

## Video Metadata

- **Title:** GreenLang Installation and Setup Guide
- **Duration:** 20 minutes
- **Level:** Intermediate
- **Audience:** System Administrators, DevOps Engineers
- **Prerequisites:** Video 01 (Introduction), Basic Docker/Kubernetes knowledge

## Learning Objectives

By the end of this video, viewers will be able to:
1. Install GreenLang using Docker Compose
2. Configure essential settings
3. Verify the installation is working correctly
4. Understand next steps for production deployment

---

## Script

### SCENE 1: Opening (0:00 - 0:30)

**VISUAL:**
- GreenLang logo animation
- Title card: "Installation and Setup"

**NARRATION:**
"Welcome back to the GreenLang tutorial series. In this video, we'll walk through the complete installation process, from preparing your environment to verifying that everything is running correctly."

**ACTION:**
- Logo animation
- Transition to title card

---

### SCENE 2: Prerequisites Overview (0:30 - 2:00)

**VISUAL:**
- Terminal window
- System requirements checklist

**NARRATION:**
"Before we begin, let's make sure your system meets the requirements.

For a development or evaluation installation, you'll need a machine with at least 4 CPU cores, 8 gigabytes of RAM, and 50 gigabytes of available storage.

You'll also need Docker version 20.10 or later, and Docker Compose version 2.0 or later.

Let's verify these are installed. Open a terminal and run docker version."

**ACTION:**
- Display requirements checklist
- Open terminal
- Type and execute: `docker --version`
- Type and execute: `docker-compose --version`

---

### SCENE 3: Download GreenLang (2:00 - 3:30)

**VISUAL:**
- Terminal with git commands
- File browser showing downloaded files

**NARRATION:**
"Now let's download GreenLang. If you have git installed, you can clone the quickstart repository directly.

Type git clone followed by the GreenLang quickstart repository URL.

Once the download completes, navigate into the directory with cd greenlang-quickstart.

Let's look at what we've downloaded. The key file is docker-compose.yml, which defines all the services we'll run. There's also an environment file template and configuration directories."

**ACTION:**
- Type: `git clone https://github.com/greenlang/greenlang-quickstart.git`
- Show download progress
- Type: `cd greenlang-quickstart`
- Type: `ls -la`
- Highlight key files

---

### SCENE 4: Configuration (3:30 - 6:00)

**VISUAL:**
- Terminal editing .env file
- Configuration explanation overlays

**NARRATION:**
"Before starting GreenLang, we need to configure a few settings.

First, copy the example environment file to create your own configuration.

Now open this file in your preferred text editor. You can use nano, vim, or any editor you're comfortable with.

Let's walk through the key settings.

GREENLANG_PORT sets the API port. The default of 8000 works for most installations.

GREENLANG_DASHBOARD_PORT sets the web dashboard port. We'll use 8080.

For security, you should change the default passwords. Set a strong password for POSTGRES_PASSWORD and REDIS_PASSWORD.

The JWT_SECRET is used for authentication tokens. Generate a random string of at least 32 characters for this.

For a development installation, the defaults are fine. But for production, you'll want to configure these carefully as outlined in our production deployment guide."

**ACTION:**
- Type: `cp .env.example .env`
- Type: `nano .env`
- Navigate through file, highlighting each important variable
- Show example of generating secure password
- Save and exit editor

---

### SCENE 5: Docker Compose File Review (6:00 - 8:00)

**VISUAL:**
- docker-compose.yml file content
- Service architecture diagram

**NARRATION:**
"Let's briefly review the Docker Compose file to understand what we're deploying.

The file defines six services.

The greenlang-api service is the main API server. It handles all REST API requests and coordinates the other services.

The greenlang-dashboard is the web interface you'll use to interact with the system.

The greenlang-agent service runs the process heat monitoring agents. In production, you might scale this to multiple instances.

The greenlang-ml service provides machine learning capabilities, including predictions and anomaly detection.

Postgres provides the main database using TimescaleDB for efficient time-series storage.

And Redis provides caching and message queuing for inter-service communication."

**ACTION:**
- Open docker-compose.yml
- Scroll through file slowly
- Overlay service architecture diagram

---

### SCENE 6: Starting the Services (8:00 - 11:00)

**VISUAL:**
- Terminal with docker-compose commands
- Container logs scrolling

**NARRATION:**
"Now we're ready to start GreenLang.

In your terminal, run docker-compose up with the -d flag to run in detached mode.

Docker will pull the required images if they're not already on your system. This may take a few minutes on the first run.

While we wait, let's check the progress. Run docker-compose ps to see the status of each container.

Once all containers show as 'running', we can check the logs to make sure everything started correctly.

Run docker-compose logs -f to follow the logs. You'll see messages from all services as they initialize.

Look for the message 'GreenLang API ready' - this indicates the API is up and accepting connections.

You can press Control-C to stop following the logs once you see the ready message."

**ACTION:**
- Type: `docker-compose up -d`
- Show image pulls completing
- Type: `docker-compose ps`
- Wait for all services to show "running"
- Type: `docker-compose logs -f`
- Highlight "GreenLang API ready" message
- Press Ctrl+C

---

### SCENE 7: Verifying the Installation (11:00 - 14:00)

**VISUAL:**
- Terminal with curl commands
- Browser showing dashboard
- Health check responses

**NARRATION:**
"Let's verify that GreenLang is running correctly.

First, check the API health endpoint. Run curl against localhost port 8000 slash health.

You should see a JSON response with status 'healthy' and the version number.

Now let's access the dashboard. Open your web browser and navigate to localhost port 8080.

You'll see the GreenLang login page. Use the default credentials: username 'admin' and password 'admin123'.

Important: In production, you should change this password immediately after your first login.

After logging in, you'll see the main dashboard. It will be mostly empty since we haven't configured any agents yet.

Let's verify the database is accessible. You can run the database check command through the API container.

And let's confirm the ML engine is running by checking its status endpoint."

**ACTION:**
- Type: `curl http://localhost:8000/health`
- Show healthy response
- Open browser, navigate to http://localhost:8080
- Show login page
- Enter credentials and login
- Show empty dashboard
- Return to terminal
- Type: `docker-compose exec greenlang-api greenlang-cli db check`
- Type: `curl http://localhost:8000/api/v1/ml/status`

---

### SCENE 8: Initial Admin Tasks (14:00 - 17:00)

**VISUAL:**
- Dashboard user settings
- Security configuration

**NARRATION:**
"With GreenLang running, there are a few administrative tasks you should complete.

First, let's change the admin password. In the dashboard, click on your username in the top right corner and select 'Profile'.

Click 'Change Password' and enter a new, secure password. Use at least 12 characters with a mix of letters, numbers, and symbols.

Next, let's set up a regular user account. Click 'Settings' in the navigation, then 'Users'.

Click 'Add User' and fill in the details. Assign the appropriate role based on their responsibilities. Operators typically get the 'operator' role, while engineers might get the 'engineer' role.

Finally, let's verify email notifications are configured. Under 'Settings', click 'Notifications'.

Enter your SMTP server details if you want to receive email alerts. This is optional but recommended for production."

**ACTION:**
- Click username, select Profile
- Show password change dialog
- Navigate to Settings > Users
- Click Add User, fill in form
- Navigate to Settings > Notifications
- Show SMTP configuration fields

---

### SCENE 9: Troubleshooting Common Issues (17:00 - 19:00)

**VISUAL:**
- Terminal showing error scenarios
- Solution overlays

**NARRATION:**
"Let's cover a few common issues you might encounter.

If containers fail to start, first check Docker is running. Then verify you have enough system resources.

If you see port conflicts, another application may be using ports 8000 or 8080. Check with netstat and either stop the conflicting application or change the ports in your .env file.

If the dashboard doesn't load, verify the dashboard container is running and check its logs for errors.

For database connection issues, make sure the postgres container is running and the password matches your environment file.

Our documentation includes a complete troubleshooting guide for more complex issues. And our community forum is a great resource if you get stuck."

**ACTION:**
- Show: `docker ps` with missing container
- Show: `netstat -an | grep 8000`
- Show: `docker-compose logs greenlang-dashboard`
- Display documentation URL

---

### SCENE 10: Closing (19:00 - 20:00)

**VISUAL:**
- Summary checklist
- Next steps

**NARRATION:**
"Congratulations! You've successfully installed GreenLang.

Let's recap what we covered: We downloaded GreenLang, configured the environment, started all services, verified the installation, and completed initial administrative tasks.

For production deployments, be sure to review our production deployment guide for high availability, security hardening, and performance optimization.

In the next video, we'll create our first process heat monitoring agent. See you there!"

**ACTION:**
- Display completion checklist with checkmarks
- Show "Next: Creating Your First Agent" card

---

## Production Notes

### Terminal Recording Requirements

- Clean terminal with readable font (16pt minimum)
- Color scheme with good contrast
- Show command prompts clearly
- Pause briefly after each command for readability

### Screen Recording Requirements

- Browser: Chrome or Firefox
- Resolution: 1920x1080
- Clear cursor visibility
- Highlight clicks

### Graphics Requirements

- Docker Compose service diagram
- System requirements checklist
- Troubleshooting flowchart

### Timing Considerations

- Allow extra time for image pulls (may vary by network speed)
- Pause for viewers to read error messages
- Clear transitions between terminal and browser

---

*Script Version: 1.0.0*
*Last Updated: December 2025*
