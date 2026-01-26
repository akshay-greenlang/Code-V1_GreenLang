
import subprocess
import sys
def get_github_token():
    try:
        input_data = "protocol=https
host=github.com

"
        result = subprocess.run(["git", "credential", "fill"], input=input_data, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split("
"):
                if line.startswith("password="):
                    return line.split("=", 1)[1]
    except Exception as e:
        pass
    return None
if __name__ == "__main__":
    token = get_github_token()
    if token:
        print(token)
    else:
        sys.exit(1)
