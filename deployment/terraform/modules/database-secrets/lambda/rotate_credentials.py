"""
GreenLang Database Credentials Rotation Lambda Function

This Lambda function handles the rotation of PostgreSQL database credentials
stored in AWS Secrets Manager. It implements the standard four-step rotation
process: createSecret, setSecret, testSecret, and finishSecret.

Author: GreenLang DevOps Team
Version: 1.0.0
"""

import json
import logging
import os
import string
import secrets
import boto3
import psycopg2
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables
SECRETS_MANAGER_ENDPOINT = os.environ.get('SECRETS_MANAGER_ENDPOINT')
DATABASE_HOST = os.environ.get('DATABASE_HOST')
DATABASE_PORT = int(os.environ.get('DATABASE_PORT', '5432'))
DATABASE_NAME = os.environ.get('DATABASE_NAME', 'greenlang')
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN')
EXCLUDED_CHARACTERS = os.environ.get('EXCLUDED_CHARACTERS', '/@"\'\\')

# Password policy
PASSWORD_LENGTH = 32
PASSWORD_SPECIAL_CHARS = '!#$%&*()-_=+[]{}<>:?'


def lambda_handler(event, context):
    """
    Main Lambda handler for secrets rotation.

    Args:
        event: Lambda event containing:
            - SecretId: ARN or name of the secret
            - ClientRequestToken: Unique identifier for the rotation request
            - Step: Current rotation step (createSecret, setSecret, testSecret, finishSecret)
        context: Lambda context object

    Returns:
        None

    Raises:
        ValueError: If required event parameters are missing or invalid
    """
    logger.info(f"Rotation event received: {json.dumps(event)}")

    # Extract event parameters
    arn = event['SecretId']
    token = event['ClientRequestToken']
    step = event['Step']

    # Create Secrets Manager client
    service_client = boto3.client(
        'secretsmanager',
        endpoint_url=SECRETS_MANAGER_ENDPOINT
    )

    # Verify the secret exists and is enabled for rotation
    metadata = service_client.describe_secret(SecretId=arn)

    if not metadata.get('RotationEnabled', False):
        logger.error(f"Secret {arn} is not enabled for rotation")
        raise ValueError(f"Secret {arn} is not enabled for rotation")

    # Verify the token is valid for this rotation
    versions = metadata.get('VersionIdsToStages', {})
    if token not in versions:
        logger.error(f"Token {token} not found in secret versions")
        raise ValueError(f"Token {token} not found in secret versions")

    # Check if rotation is already complete
    if "AWSCURRENT" in versions.get(token, []):
        logger.info(f"Token {token} is already AWSCURRENT. Rotation complete.")
        return

    # Execute the appropriate rotation step
    if step == "createSecret":
        create_secret(service_client, arn, token)
    elif step == "setSecret":
        set_secret(service_client, arn, token)
    elif step == "testSecret":
        test_secret(service_client, arn, token)
    elif step == "finishSecret":
        finish_secret(service_client, arn, token)
    else:
        raise ValueError(f"Invalid rotation step: {step}")


def create_secret(service_client, arn, token):
    """
    Create a new secret version with a new password.

    This step generates a new random password and stores it as a new
    version of the secret with the AWSPENDING staging label.

    Args:
        service_client: Secrets Manager client
        arn: Secret ARN
        token: Client request token
    """
    logger.info(f"Creating new secret version for {arn}")

    # Check if AWSPENDING version already exists
    try:
        service_client.get_secret_value(
            SecretId=arn,
            VersionId=token,
            VersionStage="AWSPENDING"
        )
        logger.info(f"AWSPENDING version already exists for token {token}")
        return
    except ClientError as e:
        if e.response['Error']['Code'] != 'ResourceNotFoundException':
            raise

    # Get current secret value
    current_secret = service_client.get_secret_value(
        SecretId=arn,
        VersionStage="AWSCURRENT"
    )
    secret_dict = json.loads(current_secret['SecretString'])

    # Generate new password
    new_password = generate_password()
    secret_dict['password'] = new_password

    # Update connection string if present
    if 'connectionString' in secret_dict:
        username = secret_dict.get('username', '')
        host = secret_dict.get('host', DATABASE_HOST)
        port = secret_dict.get('port', DATABASE_PORT)
        dbname = secret_dict.get('dbname', DATABASE_NAME)
        secret_dict['connectionString'] = f"postgresql://{username}:{new_password}@{host}:{port}/{dbname}"

    # Store new version with AWSPENDING label
    service_client.put_secret_value(
        SecretId=arn,
        ClientRequestToken=token,
        SecretString=json.dumps(secret_dict),
        VersionStages=['AWSPENDING']
    )

    logger.info(f"Successfully created new secret version for {arn}")


def set_secret(service_client, arn, token):
    """
    Set the new password in the PostgreSQL database.

    This step connects to the database using the master credentials
    and updates the password for the user specified in the secret.

    Args:
        service_client: Secrets Manager client
        arn: Secret ARN
        token: Client request token
    """
    logger.info(f"Setting secret in database for {arn}")

    # Get pending secret value
    pending_secret = service_client.get_secret_value(
        SecretId=arn,
        VersionId=token,
        VersionStage="AWSPENDING"
    )
    pending_dict = json.loads(pending_secret['SecretString'])

    # Get master credentials for database connection
    # Try to find master secret ARN from secret metadata or use default naming
    secret_metadata = service_client.describe_secret(SecretId=arn)
    secret_name = secret_metadata['Name']

    # Determine if this is the master secret itself
    is_master_secret = '/master' in secret_name

    if is_master_secret:
        # For master secret rotation, use current master credentials
        current_secret = service_client.get_secret_value(
            SecretId=arn,
            VersionStage="AWSCURRENT"
        )
        master_dict = json.loads(current_secret['SecretString'])
    else:
        # For other secrets, use master credentials
        master_secret_name = secret_name.rsplit('/', 1)[0] + '/master'
        try:
            master_secret = service_client.get_secret_value(
                SecretId=master_secret_name,
                VersionStage="AWSCURRENT"
            )
            master_dict = json.loads(master_secret['SecretString'])
        except ClientError as e:
            logger.error(f"Failed to get master credentials: {e}")
            raise

    # Connect to database and update password
    conn = None
    try:
        conn = psycopg2.connect(
            host=master_dict.get('host', DATABASE_HOST),
            port=master_dict.get('port', DATABASE_PORT),
            database=master_dict.get('dbname', DATABASE_NAME),
            user=master_dict['username'],
            password=master_dict['password'],
            connect_timeout=10
        )
        conn.autocommit = True

        with conn.cursor() as cursor:
            # Update user password
            username = pending_dict['username']
            new_password = pending_dict['password']

            # Escape password for SQL
            cursor.execute(
                "ALTER USER %s WITH PASSWORD %s",
                (psycopg2.extensions.AsIs(f'"{username}"'), new_password)
            )

            logger.info(f"Successfully updated password for user {username}")

    except Exception as e:
        logger.error(f"Failed to set password in database: {e}")
        raise
    finally:
        if conn:
            conn.close()


def test_secret(service_client, arn, token):
    """
    Test the new credentials by connecting to the database.

    This step verifies that the new credentials work by establishing
    a connection to the PostgreSQL database.

    Args:
        service_client: Secrets Manager client
        arn: Secret ARN
        token: Client request token
    """
    logger.info(f"Testing new credentials for {arn}")

    # Get pending secret value
    pending_secret = service_client.get_secret_value(
        SecretId=arn,
        VersionId=token,
        VersionStage="AWSPENDING"
    )
    pending_dict = json.loads(pending_secret['SecretString'])

    # Test connection with new credentials
    conn = None
    try:
        conn = psycopg2.connect(
            host=pending_dict.get('host', DATABASE_HOST),
            port=pending_dict.get('port', DATABASE_PORT),
            database=pending_dict.get('dbname', DATABASE_NAME),
            user=pending_dict['username'],
            password=pending_dict['password'],
            connect_timeout=10
        )

        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()

            if result[0] != 1:
                raise ValueError("Test query returned unexpected result")

        logger.info(f"Successfully tested new credentials for {arn}")

    except Exception as e:
        logger.error(f"Failed to test new credentials: {e}")
        raise
    finally:
        if conn:
            conn.close()


def finish_secret(service_client, arn, token):
    """
    Finish the rotation by promoting the new secret version.

    This step moves the AWSCURRENT label from the old version to
    the new version and marks the old version as AWSPREVIOUS.

    Args:
        service_client: Secrets Manager client
        arn: Secret ARN
        token: Client request token
    """
    logger.info(f"Finishing rotation for {arn}")

    # Get current version info
    metadata = service_client.describe_secret(SecretId=arn)
    versions = metadata.get('VersionIdsToStages', {})

    # Find current version
    current_version = None
    for version_id, stages in versions.items():
        if "AWSCURRENT" in stages:
            if version_id == token:
                logger.info(f"Token {token} is already AWSCURRENT. Nothing to do.")
                return
            current_version = version_id
            break

    # Move AWSCURRENT to the new version
    service_client.update_secret_version_stage(
        SecretId=arn,
        VersionStage="AWSCURRENT",
        MoveToVersionId=token,
        RemoveFromVersionId=current_version
    )

    logger.info(f"Successfully rotated secret {arn} to version {token}")

    # Send notification
    send_rotation_notification(arn, token)


def generate_password():
    """
    Generate a secure random password.

    Returns:
        str: A randomly generated password meeting the password policy
    """
    # Character sets
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    special = PASSWORD_SPECIAL_CHARS

    # Remove excluded characters
    for char in EXCLUDED_CHARACTERS:
        lowercase = lowercase.replace(char, '')
        uppercase = uppercase.replace(char, '')
        digits = digits.replace(char, '')
        special = special.replace(char, '')

    # Ensure password contains at least one of each type
    password = [
        secrets.choice(lowercase),
        secrets.choice(uppercase),
        secrets.choice(digits),
        secrets.choice(special)
    ]

    # Fill remaining length with random characters from all sets
    all_chars = lowercase + uppercase + digits + special
    password.extend(secrets.choice(all_chars) for _ in range(PASSWORD_LENGTH - 4))

    # Shuffle the password
    password_list = list(password)
    secrets.SystemRandom().shuffle(password_list)

    return ''.join(password_list)


def send_rotation_notification(arn, token):
    """
    Send a notification that secret rotation has completed.

    Args:
        arn: Secret ARN
        token: Client request token
    """
    if not SNS_TOPIC_ARN:
        logger.info("No SNS topic configured, skipping notification")
        return

    try:
        sns_client = boto3.client('sns')

        message = {
            "event": "SecretRotationCompleted",
            "secretArn": arn,
            "versionId": token,
            "timestamp": str(boto3.client('sts').get_caller_identity()['Account'])
        }

        sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject="GreenLang Database Secret Rotation Completed",
            Message=json.dumps(message, indent=2),
            MessageAttributes={
                "EventType": {
                    "DataType": "String",
                    "StringValue": "SecretRotationCompleted"
                },
                "SecretArn": {
                    "DataType": "String",
                    "StringValue": arn
                }
            }
        )

        logger.info(f"Rotation notification sent to {SNS_TOPIC_ARN}")

    except Exception as e:
        logger.warning(f"Failed to send rotation notification: {e}")
        # Don't fail the rotation if notification fails


def get_secret_dict(service_client, arn, stage, token=None):
    """
    Helper function to get and parse a secret value.

    Args:
        service_client: Secrets Manager client
        arn: Secret ARN
        stage: Version stage (AWSCURRENT, AWSPENDING, etc.)
        token: Optional version ID

    Returns:
        dict: Parsed secret value
    """
    kwargs = {
        'SecretId': arn,
        'VersionStage': stage
    }
    if token:
        kwargs['VersionId'] = token

    response = service_client.get_secret_value(**kwargs)
    return json.loads(response['SecretString'])
