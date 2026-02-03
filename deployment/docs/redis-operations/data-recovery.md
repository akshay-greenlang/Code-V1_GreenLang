# Redis Data Recovery Guide

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Last Updated | 2026-02-03 |
| Owner | DevOps Team |
| Purpose | Data recovery procedures for Redis |

---

## 1. Overview

This guide provides detailed procedures for recovering Redis data from various backup types and failure scenarios. Redis supports two persistence mechanisms:

- **RDB (Redis Database)**: Point-in-time snapshots
- **AOF (Append-Only File)**: Transaction log of all write operations

### Recovery Decision Matrix

| Scenario | Recommended Recovery Method |
|----------|---------------------------|
| Complete data loss | RDB restore (fastest) |
| Corruption detected | AOF repair + restore |
| Point-in-time recovery needed | AOF truncation |
| Partial data recovery | Key extraction from backup |
| Accidental deletion | Recent RDB + AOF replay |

---

## 2. Understanding Redis Persistence

### RDB (Snapshots)

```
RDB SNAPSHOT PROCESS:

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Redis       │     │    Fork      │     │   dump.rdb   │
│  Memory      │ ──► │   Process    │ ──► │   (on disk)  │
│              │     │   (child)    │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                     Compressed binary
                     snapshot of all data
```

**RDB Configuration**:
```conf
# Save snapshot every 900 seconds if at least 1 key changed
save 900 1
# Save every 300 seconds if at least 10 keys changed
save 300 10
# Save every 60 seconds if at least 10000 keys changed
save 60 10000

# Compress RDB file
rdbcompression yes

# Checksum for corruption detection
rdbchecksum yes

# RDB filename
dbfilename dump.rdb

# Directory
dir /var/lib/redis
```

### AOF (Append-Only File)

```
AOF WRITE PROCESS:

Client         Redis              AOF Buffer           AOF File
  │              │                    │                   │
  │──WRITE───────►                    │                   │
  │              │──append command────►                   │
  │              │                    │──fsync (based on─►│
  │              │                    │   policy)         │
  │◄─────────OK──│                    │                   │
```

**AOF Configuration**:
```conf
# Enable AOF
appendonly yes

# AOF filename
appendfilename "appendonly.aof"

# Sync policy:
# always: fsync after every write (safest, slowest)
# everysec: fsync every second (good balance)
# no: let OS handle it (fastest, risky)
appendfsync everysec

# Rewrite AOF when it grows 100% from last rewrite
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
```

---

## 3. RDB Restore Procedure

### 3.1 Full RDB Restore

**Prerequisites**:
- Valid RDB backup file
- Redis service stopped
- Sufficient disk space

```bash
#!/bin/bash
# rdb-restore.sh - Full RDB restore procedure

REDIS_DATA_DIR="/var/lib/redis"
BACKUP_SOURCE="s3://greenlang-backups/redis"
BACKUP_FILE="dump-20260203-120000.rdb"

echo "=== RDB Restore Procedure ==="
echo "Backup file: $BACKUP_FILE"

# Step 1: Stop Redis
echo -e "\n[1/7] Stopping Redis..."
systemctl stop redis
sleep 5

# Step 2: Verify Redis is stopped
if pgrep -x redis-server > /dev/null; then
  echo "ERROR: Redis is still running"
  exit 1
fi

# Step 3: Backup current data
echo -e "\n[2/7] Backing up current data..."
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p /var/lib/redis/backup-$TIMESTAMP
cp $REDIS_DATA_DIR/dump.rdb /var/lib/redis/backup-$TIMESTAMP/ 2>/dev/null
cp $REDIS_DATA_DIR/appendonly.aof /var/lib/redis/backup-$TIMESTAMP/ 2>/dev/null

# Step 4: Download backup
echo -e "\n[3/7] Downloading backup from S3..."
aws s3 cp $BACKUP_SOURCE/$BACKUP_FILE /tmp/restore-dump.rdb

# Step 5: Verify backup integrity
echo -e "\n[4/7] Verifying backup integrity..."
redis-check-rdb /tmp/restore-dump.rdb
if [ $? -ne 0 ]; then
  echo "ERROR: RDB file is corrupted"
  exit 1
fi

# Step 6: Copy backup to Redis data directory
echo -e "\n[5/7] Restoring RDB file..."
cp /tmp/restore-dump.rdb $REDIS_DATA_DIR/dump.rdb
chown redis:redis $REDIS_DATA_DIR/dump.rdb
chmod 640 $REDIS_DATA_DIR/dump.rdb

# Step 7: Remove AOF if exists (will be regenerated)
echo -e "\n[6/7] Handling AOF..."
if [ -f "$REDIS_DATA_DIR/appendonly.aof" ]; then
  mv $REDIS_DATA_DIR/appendonly.aof $REDIS_DATA_DIR/appendonly.aof.old
fi

# Step 8: Start Redis
echo -e "\n[7/7] Starting Redis..."
systemctl start redis
sleep 10

# Verify
echo -e "\n=== Verification ==="
redis-cli PING
if [ $? -eq 0 ]; then
  DBSIZE=$(redis-cli DBSIZE)
  echo "Redis restored successfully"
  echo "Database size: $DBSIZE"

  # Regenerate AOF if it was enabled
  redis-cli CONFIG GET appendonly | grep -q "yes"
  if [ $? -eq 0 ]; then
    echo "Regenerating AOF from RDB..."
    redis-cli BGREWRITEAOF
  fi
else
  echo "ERROR: Redis failed to start"
  exit 1
fi

# Cleanup
rm /tmp/restore-dump.rdb
echo -e "\n=== Restore Complete ==="
```

### 3.2 Verify RDB Before Restore

```bash
# Check RDB file integrity
redis-check-rdb dump.rdb

# Expected output for valid file:
# [offset 0] Checking RDB file dump.rdb
# [offset 26] AUX FIELD redis-ver = '7.0.0'
# [offset xxx] \o/ RDB looks OK! \o/
# [info] xxx keys read
# [info] xxx expires

# If corrupted:
# [offset xxx] !!! RDB file corrupted !!!
# Unexpected end of file reading xxx at xxx
```

### 3.3 Restore to Specific Time (Multiple RDB Files)

```bash
#!/bin/bash
# List available backups
aws s3 ls s3://greenlang-backups/redis/ --recursive | sort

# Example output:
# 2026-02-03 06:00:00 dump-20260203-060000.rdb
# 2026-02-03 12:00:00 dump-20260203-120000.rdb
# 2026-02-03 18:00:00 dump-20260203-180000.rdb

# Restore specific point in time
# If incident at 14:00, restore 12:00 backup
aws s3 cp s3://greenlang-backups/redis/dump-20260203-120000.rdb /tmp/
# Then follow full restore procedure
```

---

## 4. AOF Recovery

### 4.1 AOF Repair and Recovery

```bash
#!/bin/bash
# aof-recovery.sh - AOF repair and recovery procedure

AOF_FILE="/var/lib/redis/appendonly.aof"

echo "=== AOF Recovery Procedure ==="

# Step 1: Check AOF validity
echo -e "\n[1/5] Checking AOF validity..."
redis-check-aof $AOF_FILE
STATUS=$?

if [ $STATUS -eq 0 ]; then
  echo "AOF file is valid"
  exit 0
fi

# Step 2: Backup original AOF
echo -e "\n[2/5] Backing up original AOF..."
cp $AOF_FILE ${AOF_FILE}.bak-$(date +%Y%m%d-%H%M%S)

# Step 3: Analyze corruption
echo -e "\n[3/5] Analyzing corruption..."
redis-check-aof $AOF_FILE 2>&1 | head -20

# Step 4: Attempt fix
echo -e "\n[4/5] Attempting automatic fix..."
echo "This will truncate the AOF at the corruption point."
echo "Some recent data may be lost."
read -p "Continue? (yes/no): " confirm

if [ "$confirm" == "yes" ]; then
  redis-check-aof --fix $AOF_FILE

  if [ $? -eq 0 ]; then
    echo "AOF repaired successfully"
  else
    echo "ERROR: Could not repair AOF"
    exit 1
  fi
fi

# Step 5: Restart Redis with repaired AOF
echo -e "\n[5/5] Restarting Redis..."
systemctl restart redis
sleep 10

# Verify
redis-cli PING
if [ $? -eq 0 ]; then
  DBSIZE=$(redis-cli DBSIZE)
  echo "Redis restored from repaired AOF"
  echo "Database size: $DBSIZE"
else
  echo "ERROR: Redis failed to start"
  echo "Consider restoring from RDB backup instead"
  exit 1
fi
```

### 4.2 Manual AOF Truncation

For precise control over where to truncate:

```bash
# Find the corruption point
redis-check-aof /var/lib/redis/appendonly.aof 2>&1 | grep -E "(offset|Unexpected)"
# Example: "Unexpected end of file reading SET at 12345678"

# Manually truncate at safe point
truncate -s 12345600 /var/lib/redis/appendonly.aof

# Verify truncated file
redis-check-aof /var/lib/redis/appendonly.aof
```

### 4.3 AOF Rewrite to Clean Up

After recovery, rewrite AOF to optimize:

```bash
# Trigger AOF rewrite
redis-cli BGREWRITEAOF

# Monitor progress
watch -n 1 'redis-cli INFO persistence | grep aof_rewrite_in_progress'

# After completion, check new AOF size
ls -lh /var/lib/redis/appendonly.aof
```

---

## 5. Partial Data Recovery

### 5.1 Recover Specific Keys from RDB

```bash
#!/bin/bash
# recover-keys.sh - Extract specific keys from RDB backup

BACKUP_RDB="/tmp/dump-backup.rdb"
PRODUCTION_HOST="redis-master.greenlang.internal"
KEY_PATTERN="user:*"

echo "=== Partial Key Recovery ==="

# Step 1: Start temporary Redis with backup
echo "[1/4] Starting temporary Redis instance..."
docker run -d --name redis-recovery \
  -p 16379:6379 \
  -v $BACKUP_RDB:/data/dump.rdb \
  redis:7-alpine

sleep 5

# Step 2: Extract matching keys
echo -e "\n[2/4] Extracting keys matching pattern: $KEY_PATTERN"
KEYS=$(redis-cli -p 16379 --scan --pattern "$KEY_PATTERN")
KEY_COUNT=$(echo "$KEYS" | wc -l)
echo "Found $KEY_COUNT keys"

# Step 3: Dump and restore each key
echo -e "\n[3/4] Restoring keys to production..."
echo "$KEYS" | while read key; do
  if [ -n "$key" ]; then
    # Dump from backup instance
    DUMP=$(redis-cli -p 16379 DUMP "$key" 2>/dev/null)
    TTL=$(redis-cli -p 16379 PTTL "$key" 2>/dev/null)

    if [ -n "$DUMP" ] && [ "$DUMP" != "" ]; then
      # Set TTL to 0 if key has no expiry
      [ "$TTL" -lt 0 ] && TTL=0

      # Restore to production (REPLACE if exists)
      redis-cli -h $PRODUCTION_HOST RESTORE "$key" $TTL "$DUMP" REPLACE 2>/dev/null

      if [ $? -eq 0 ]; then
        echo "Restored: $key"
      else
        echo "Failed: $key"
      fi
    fi
  fi
done

# Step 4: Cleanup
echo -e "\n[4/4] Cleaning up..."
docker stop redis-recovery && docker rm redis-recovery

echo -e "\n=== Recovery Complete ==="
```

### 5.2 Recover Keys by Type

```bash
#!/bin/bash
# recover-by-type.sh - Recover keys of specific type

recover_strings() {
  local key=$1
  local value=$(redis-cli -p 16379 GET "$key")
  redis-cli -h $PRODUCTION_HOST SET "$key" "$value"
}

recover_hash() {
  local key=$1
  redis-cli -p 16379 HGETALL "$key" | xargs -n 2 sh -c \
    "redis-cli -h $PRODUCTION_HOST HSET '$key' \"\$0\" \"\$1\""
}

recover_list() {
  local key=$1
  redis-cli -p 16379 LRANGE "$key" 0 -1 | while read value; do
    redis-cli -h $PRODUCTION_HOST RPUSH "$key" "$value"
  done
}

recover_set() {
  local key=$1
  redis-cli -p 16379 SMEMBERS "$key" | while read value; do
    redis-cli -h $PRODUCTION_HOST SADD "$key" "$value"
  done
}

recover_zset() {
  local key=$1
  redis-cli -p 16379 ZRANGE "$key" 0 -1 WITHSCORES | paste - - | while read member score; do
    redis-cli -h $PRODUCTION_HOST ZADD "$key" "$score" "$member"
  done
}

# Main recovery logic
KEY=$1
TYPE=$(redis-cli -p 16379 TYPE "$KEY" | tr -d '\r')

case $TYPE in
  string) recover_strings "$KEY" ;;
  hash)   recover_hash "$KEY" ;;
  list)   recover_list "$KEY" ;;
  set)    recover_set "$KEY" ;;
  zset)   recover_zset "$KEY" ;;
  *)      echo "Unknown type: $TYPE" ;;
esac
```

---

## 6. Point-in-Time Recovery Using AOF

### 6.1 Understanding AOF Structure

```
AOF FILE STRUCTURE:

*3               <- Number of arguments (3 for SET key value)
$3               <- Length of first argument (3 chars)
SET              <- Command
$4               <- Length of second argument
user             <- Key
$5               <- Length of third argument
alice            <- Value

EXAMPLE AOF CONTENT:
*2
$6
SELECT
$1
0
*3
$3
SET
$8
user:100
$5
alice
*3
$3
SET
$8
user:101
$3
bob
```

### 6.2 Point-in-Time Recovery Steps

```bash
#!/bin/bash
# pit-recovery.sh - Point-in-time recovery using AOF

AOF_FILE="/var/lib/redis/appendonly.aof"
RECOVERY_TIME="2026-02-03 14:30:00"

echo "=== Point-in-Time Recovery ==="
echo "Target time: $RECOVERY_TIME"

# Step 1: Convert time to Unix timestamp
TARGET_TS=$(date -d "$RECOVERY_TIME" +%s)
echo "Target timestamp: $TARGET_TS"

# Step 2: Backup current AOF
echo -e "\n[1/5] Backing up current AOF..."
cp $AOF_FILE ${AOF_FILE}.pit-backup

# Step 3: Find commands after target time
# This requires UNIX TIMESTAMP comments in AOF (Redis 7.0+)
# Or correlation with other logs

echo -e "\n[2/5] Analyzing AOF timestamps..."
# For Redis 7.0+ with timestamp annotations
grep -n "^#TS:" $AOF_FILE | while read line; do
  LINE_NUM=$(echo $line | cut -d: -f1)
  TIMESTAMP=$(echo $line | cut -d: -f3)
  if [ "$TIMESTAMP" -gt "$TARGET_TS" ]; then
    echo "First command after target time at line: $LINE_NUM"
    echo $LINE_NUM > /tmp/truncate_line
    break
  fi
done

# Step 4: Truncate AOF at recovery point
if [ -f /tmp/truncate_line ]; then
  TRUNCATE_LINE=$(cat /tmp/truncate_line)
  echo -e "\n[3/5] Truncating AOF at line $TRUNCATE_LINE..."
  head -n $((TRUNCATE_LINE - 1)) $AOF_FILE > ${AOF_FILE}.recovered
  mv ${AOF_FILE}.recovered $AOF_FILE
fi

# Step 5: Verify and restart
echo -e "\n[4/5] Verifying truncated AOF..."
redis-check-aof $AOF_FILE

echo -e "\n[5/5] Restarting Redis..."
systemctl restart redis

# Cleanup
rm /tmp/truncate_line 2>/dev/null

echo -e "\n=== Recovery Complete ==="
```

### 6.3 Alternative: AOF Replay with Filtering

```python
#!/usr/bin/env python3
"""
aof_filter.py - Filter AOF to remove specific commands

Usage: python aof_filter.py input.aof output.aof --exclude-keys "user:*" --after "2026-02-03 14:30:00"
"""

import argparse
import re
import sys
from datetime import datetime

def parse_aof_command(lines, start_idx):
    """Parse a single Redis command from AOF format"""
    if start_idx >= len(lines):
        return None, start_idx

    line = lines[start_idx]
    if not line.startswith('*'):
        return None, start_idx + 1

    num_args = int(line[1:])
    command_parts = []
    idx = start_idx + 1

    for _ in range(num_args):
        if idx >= len(lines):
            return None, idx

        # Skip length indicator ($N)
        if lines[idx].startswith('$'):
            idx += 1

        if idx < len(lines):
            command_parts.append(lines[idx])
            idx += 1

    return command_parts, idx

def should_exclude(command_parts, exclude_patterns, exclude_commands):
    """Check if command should be excluded"""
    if not command_parts:
        return False

    cmd = command_parts[0].upper()

    # Check excluded commands
    if cmd in exclude_commands:
        return True

    # Check excluded key patterns
    if len(command_parts) > 1:
        key = command_parts[1]
        for pattern in exclude_patterns:
            if re.match(pattern.replace('*', '.*'), key):
                return True

    return False

def filter_aof(input_file, output_file, exclude_keys=None, exclude_commands=None):
    """Filter AOF file"""
    exclude_patterns = exclude_keys or []
    exclude_cmds = set(cmd.upper() for cmd in (exclude_commands or []))

    with open(input_file, 'r') as f:
        lines = [line.rstrip('\n\r') for line in f]

    output_lines = []
    idx = 0
    excluded_count = 0
    total_count = 0

    while idx < len(lines):
        start_idx = idx
        command_parts, idx = parse_aof_command(lines, idx)

        if command_parts:
            total_count += 1
            if should_exclude(command_parts, exclude_patterns, exclude_cmds):
                excluded_count += 1
            else:
                # Keep this command
                output_lines.extend(lines[start_idx:idx])

    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"Total commands: {total_count}")
    print(f"Excluded: {excluded_count}")
    print(f"Remaining: {total_count - excluded_count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input AOF file')
    parser.add_argument('output', help='Output AOF file')
    parser.add_argument('--exclude-keys', nargs='+', help='Key patterns to exclude')
    parser.add_argument('--exclude-commands', nargs='+', help='Commands to exclude')

    args = parser.parse_args()
    filter_aof(args.input, args.output, args.exclude_keys, args.exclude_commands)
```

---

## 7. Recovery Validation

### 7.1 Data Integrity Checks

```bash
#!/bin/bash
# validate-recovery.sh - Post-recovery validation

echo "=== Post-Recovery Validation ==="

# 1. Basic connectivity
echo -e "\n[1/6] Connectivity check..."
redis-cli PING

# 2. Database size
echo -e "\n[2/6] Database size..."
DBSIZE=$(redis-cli DBSIZE)
echo "Keys: $DBSIZE"

# 3. Memory usage
echo -e "\n[3/6] Memory usage..."
redis-cli INFO memory | grep -E "(used_memory_human|used_memory_peak_human)"

# 4. Key type distribution
echo -e "\n[4/6] Key type distribution..."
echo "Strings: $(redis-cli --scan | head -1000 | while read k; do redis-cli TYPE "$k"; done | grep -c string)"
echo "Hashes: $(redis-cli --scan | head -1000 | while read k; do redis-cli TYPE "$k"; done | grep -c hash)"
echo "Lists: $(redis-cli --scan | head -1000 | while read k; do redis-cli TYPE "$k"; done | grep -c list)"
echo "Sets: $(redis-cli --scan | head -1000 | while read k; do redis-cli TYPE "$k"; done | grep -c set)"

# 5. Sample data verification
echo -e "\n[5/6] Sample data verification..."
echo "Checking 5 random keys..."
redis-cli --scan | head -5 | while read key; do
  TYPE=$(redis-cli TYPE "$key")
  TTL=$(redis-cli TTL "$key")
  echo "  $key (type: $TYPE, ttl: $TTL)"
done

# 6. Write test
echo -e "\n[6/6] Write test..."
TEST_KEY="_recovery_test_$(date +%s)"
redis-cli SET $TEST_KEY "test_value" > /dev/null
RESULT=$(redis-cli GET $TEST_KEY)
redis-cli DEL $TEST_KEY > /dev/null

if [ "$RESULT" == "test_value" ]; then
  echo "Write test: PASSED"
else
  echo "Write test: FAILED"
fi

echo -e "\n=== Validation Complete ==="
```

### 7.2 Compare with Pre-Incident State

```bash
#!/bin/bash
# compare-state.sh - Compare current state with pre-incident baseline

PRE_INCIDENT_DBSIZE=125000
PRE_INCIDENT_KEYS_FILE="/tmp/pre_incident_keys.txt"

echo "=== State Comparison ==="

# Get current state
CURRENT_DBSIZE=$(redis-cli DBSIZE | awk '{print $2}')
echo "Pre-incident DBSIZE: $PRE_INCIDENT_DBSIZE"
echo "Current DBSIZE: $CURRENT_DBSIZE"
echo "Difference: $((CURRENT_DBSIZE - PRE_INCIDENT_DBSIZE))"

# If we have a pre-incident key list
if [ -f "$PRE_INCIDENT_KEYS_FILE" ]; then
  echo -e "\nChecking for missing keys..."

  MISSING=0
  while read key; do
    EXISTS=$(redis-cli EXISTS "$key")
    if [ "$EXISTS" -eq 0 ]; then
      echo "Missing: $key"
      ((MISSING++))
    fi
  done < "$PRE_INCIDENT_KEYS_FILE"

  echo "Total missing keys: $MISSING"
fi
```

---

## 8. Backup Best Practices

### 8.1 Backup Strategy

```
RECOMMENDED BACKUP SCHEDULE:

┌─────────────────────────────────────────────────────────────┐
│                    BACKUP STRATEGY                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Continuous:  AOF with everysec fsync                       │
│               └── RPO: ~1 second                            │
│                                                             │
│  Hourly:      RDB snapshot to S3                            │
│               └── Retention: 24 hours                       │
│                                                             │
│  Daily:       RDB snapshot to S3                            │
│               └── Retention: 30 days                        │
│                                                             │
│  Weekly:      RDB snapshot to S3 + Glacier                  │
│               └── Retention: 1 year                         │
│                                                             │
│  Monthly:     RDB snapshot to Glacier Deep Archive          │
│               └── Retention: 7 years (compliance)           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Automated Backup Script

```bash
#!/bin/bash
# redis-backup.sh - Automated backup script

REDIS_HOST="localhost"
REDIS_PORT="6379"
BACKUP_DIR="/var/lib/redis/backups"
S3_BUCKET="s3://greenlang-backups/redis"
RETENTION_DAYS=7

# Trigger RDB save
redis-cli -h $REDIS_HOST -p $REDIS_PORT BGSAVE

# Wait for save to complete
while [ $(redis-cli -h $REDIS_HOST -p $REDIS_PORT LASTSAVE) == $(cat /tmp/last_save 2>/dev/null) ]; do
  sleep 1
done
redis-cli -h $REDIS_HOST -p $REDIS_PORT LASTSAVE > /tmp/last_save

# Copy RDB with timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
cp /var/lib/redis/dump.rdb $BACKUP_DIR/dump-$TIMESTAMP.rdb

# Upload to S3
aws s3 cp $BACKUP_DIR/dump-$TIMESTAMP.rdb $S3_BUCKET/dump-$TIMESTAMP.rdb

# Cleanup old local backups
find $BACKUP_DIR -name "dump-*.rdb" -mtime +$RETENTION_DAYS -delete

echo "Backup completed: dump-$TIMESTAMP.rdb"
```

---

## Appendix: Quick Reference

### Recovery Commands

| Scenario | Command |
|----------|---------|
| Check RDB | `redis-check-rdb dump.rdb` |
| Check AOF | `redis-check-aof appendonly.aof` |
| Fix AOF | `redis-check-aof --fix appendonly.aof` |
| Trigger RDB save | `redis-cli BGSAVE` |
| Trigger AOF rewrite | `redis-cli BGREWRITEAOF` |
| Get last save time | `redis-cli LASTSAVE` |
| Dump key (for backup) | `redis-cli DUMP keyname` |
| Restore key | `redis-cli RESTORE keyname ttl serialized-value` |

### Recovery Time Estimates

| Operation | Estimated Time |
|-----------|---------------|
| RDB restore (1GB) | 30-60 seconds |
| RDB restore (10GB) | 5-10 minutes |
| AOF replay (1M commands) | 2-5 minutes |
| AOF repair | 1-5 minutes |
| Key-by-key recovery (1000 keys) | 1-2 minutes |
