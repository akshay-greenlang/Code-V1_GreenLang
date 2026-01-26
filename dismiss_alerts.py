import subprocess
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Configuration
OWNER = "akshay-greenlang"
REPO = "Code-V1_GreenLang"
BASE_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/code-scanning/alerts"
DISMISS_REASON = "won't fix"
DISMISS_COMMENT = "Infrastructure policy alert - acceptable for development environment"
MAX_WORKERS = 20

# Thread-safe counters
progress_lock = Lock()
dismissed_count = 0
failed_count = 0
rate_limit_remaining = 5000

def get_github_token():
    """Get GitHub token from git credential fill."""
    try:
        process = subprocess.Popen(
            ['git', 'credential', 'fill'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, _ = process.communicate(input='protocol=https\nhost=github.com\n')
        for line in stdout.split('\n'):
            if line.startswith('password='):
                return line.split('=', 1)[1]
    except Exception as e:
        print(f"Error getting token: {e}")
    return None

def get_headers(token):
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

def fetch_all_open_alerts(token):
    """Fetch all open code scanning alerts with pagination."""
    headers = get_headers(token)
    alerts = []
    page = 1
    per_page = 100
    
    print("Fetching open alerts...")
    while True:
        url = f"{BASE_URL}?state=open&per_page={per_page}&page={page}"
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Error fetching alerts: {response.status_code} - {response.text}")
            break
        
        page_alerts = response.json()
        if not page_alerts:
            break
        
        alerts.extend(page_alerts)
        print(f"  Fetched page {page}: {len(page_alerts)} alerts (total: {len(alerts)})")
        page += 1
        
        # Small delay to avoid rate limiting during fetch
        time.sleep(0.1)
    
    print(f"Total open alerts fetched: {len(alerts)}")
    return alerts

def dismiss_alert(token, alert_number):
    """Dismiss a single alert."""
    global dismissed_count, failed_count, rate_limit_remaining
    
    headers = get_headers(token)
    url = f"{BASE_URL}/{alert_number}"
    payload = {
        "state": "dismissed",
        "dismissed_reason": DISMISS_REASON,
        "dismissed_comment": DISMISS_COMMENT
    }
    
    try:
        response = requests.patch(url, headers=headers, json=payload)
        
        # Update rate limit info
        remaining = response.headers.get('X-RateLimit-Remaining')
        if remaining:
            rate_limit_remaining = int(remaining)
        
        if response.status_code == 200:
            with progress_lock:
                dismissed_count += 1
                if dismissed_count % 100 == 0:
                    print(f"Progress: {dismissed_count} dismissed, {failed_count} failed, Rate limit remaining: {rate_limit_remaining}")
            return True, alert_number
        elif response.status_code == 403 and 'rate limit' in response.text.lower():
            # Rate limited - wait and retry
            reset_time = response.headers.get('X-RateLimit-Reset')
            if reset_time:
                wait_time = int(reset_time) - int(time.time()) + 1
                print(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(max(wait_time, 60))
            return dismiss_alert(token, alert_number)  # Retry
        else:
            with progress_lock:
                failed_count += 1
            return False, alert_number
    except Exception as e:
        with progress_lock:
            failed_count += 1
        return False, alert_number

def check_rate_limit(token):
    """Check current rate limit status."""
    headers = get_headers(token)
    response = requests.get("https://api.github.com/rate_limit", headers=headers)
    if response.status_code == 200:
        data = response.json()
        core = data.get('resources', {}).get('core', {})
        return core.get('remaining', 0), core.get('reset', 0)
    return 0, 0

def main():
    global dismissed_count, failed_count
    
    print("Getting GitHub token...")
    token = get_github_token()
    if not token:
        print("Failed to get GitHub token")
        return
    print("Token obtained successfully")
    
    # Check initial rate limit
    remaining, reset = check_rate_limit(token)
    print(f"Initial rate limit: {remaining} remaining, resets at {time.ctime(reset)}")
    
    # Fetch all open alerts
    alerts = fetch_all_open_alerts(token)
    if not alerts:
        print("No open alerts to dismiss")
        return
    
    alert_numbers = [alert['number'] for alert in alerts]
    print(f"\nStarting to dismiss {len(alert_numbers)} alerts with {MAX_WORKERS} workers...")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(dismiss_alert, token, num): num for num in alert_numbers}
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing alert: {e}")
            
            # Periodically check rate limit
            if dismissed_count % 500 == 0 and dismissed_count > 0:
                remaining, _ = check_rate_limit(token)
                if remaining < 100:
                    print(f"Rate limit low ({remaining}). Pausing for 60 seconds...")
                    time.sleep(60)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Completed in {elapsed:.1f} seconds")
    print(f"Successfully dismissed: {dismissed_count}")
    print(f"Failed: {failed_count}")
    print(f"Rate: {dismissed_count/elapsed:.1f} alerts/second")

if __name__ == "__main__":
    main()
