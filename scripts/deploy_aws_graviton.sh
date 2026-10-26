#!/usr/bin/env bash
# ==============================================================================
# AeroGuardian AWS Graviton Deployment Script
# Provisions an ARM64 EC2 instance, configures security groups, and runs
# AeroGuardian on AWS Graviton using the isolated [aeroguardian] IAM profile.
# ==============================================================================

set -euo pipefail

PROFILE="aeroguardian"
REGION="${AWS_REGION:-ap-south-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t4g.medium}"
SG_NAME="aeroguardian-security-group"
KEY_NAME="aeroguardian-key"

echo "=== AeroGuardian AWS Graviton Deployment ==="
echo "AWS Profile: $PROFILE"
echo "Region:      $REGION"
echo "Instance:    $INSTANCE_TYPE (ARM64 Graviton)"

# 1. Verify AWS Identity
echo "[1/6] Verifying AWS IAM caller identity..."
CALLER_JSON=$(aws sts get-caller-identity --profile "$PROFILE" --output json)
ACCOUNT_ID=$(echo "$CALLER_JSON" | grep -o '"Account": "[^"]*' | cut -d'"' -f4)
ARN=$(echo "$CALLER_JSON" | grep -o '"Arn": "[^"]*' | cut -d'"' -f4)
echo "      Authenticated as: $ARN (Account: $ACCOUNT_ID)"

# 2. Get Default VPC
echo "[2/6] Detecting default VPC in $REGION..."
VPC_ID=$(aws ec2 describe-vpcs --profile "$PROFILE" --region "$REGION" \
    --filters "Name=isDefault,Values=true" \
    --query "Vpcs[0].VpcId" --output text)

if [ "$VPC_ID" == "None" ] || [ -z "$VPC_ID" ]; then
    echo "ERROR: No default VPC found in $REGION. Please specify a VPC ID."
    exit 1
fi
echo "      Using VPC: $VPC_ID"

# 3. Create / Retrieve Security Group
echo "[3/6] Setting up security group ($SG_NAME)..."
SG_ID=$(aws ec2 describe-security-groups --profile "$PROFILE" --region "$REGION" \
    --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || true)

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
    SG_ID=$(aws ec2 create-security-group --profile "$PROFILE" --region "$REGION" \
        --group-name "$SG_NAME" \
        --description "AeroGuardian Tactical HUD and MAVLink UDP Gateway" \
        --vpc-id "$VPC_ID" \
        --query "GroupId" --output text)
    echo "      Created Security Group: $SG_ID"

    # Authorize Inbound Ports:
    # 22/tcp (SSH administration)
    aws ec2 authorize-security-group-ingress --profile "$PROFILE" --region "$REGION" \
        --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0 >/dev/null
    # 8000/tcp (AeroGuardian Tactical HUD Dashboard)
    aws ec2 authorize-security-group-ingress --profile "$PROFILE" --region "$REGION" \
        --group-id "$SG_ID" --protocol tcp --port 8000 --cidr 0.0.0.0/0 >/dev/null
    # 14550/udp (MAVLink 2.0 UDP GCS Telemetry Socket)
    aws ec2 authorize-security-group-ingress --profile "$PROFILE" --region "$REGION" \
        --group-id "$SG_ID" --protocol udp --port 14550 --cidr 0.0.0.0/0 >/dev/null
    echo "      Authorized ports: 22/tcp, 8000/tcp (HUD), 14550/udp (MAVLink)"
else
    echo "      Reusing existing Security Group: $SG_ID"
fi

# 4. Find Latest Ubuntu 24.04 LTS ARM64 AMI
echo "[4/6] Querying official Canonical Ubuntu 24.04 LTS ARM64 AMI..."
AMI_ID=$(aws ec2 describe-images --profile "$PROFILE" --region "$REGION" \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-arm64-server-*" \
              "Name=state,Values=available" \
    --query "reverse(sort_by(Images, &CreationDate))[0].ImageId" --output text)
echo "      AMI ID: $AMI_ID"

# 5. Create Key Pair if not already present
echo "[5/6] Checking SSH key pair..."
KEY_EXISTS=$(aws ec2 describe-key-pairs --profile "$PROFILE" --region "$REGION" \
    --filters "Name=key-name,Values=$KEY_NAME" \
    --query "KeyPairs[0].KeyName" --output text 2>/dev/null || true)

if [ "$KEY_EXISTS" == "None" ] || [ -z "$KEY_EXISTS" ]; then
    echo "      Generating SSH key pair $KEY_NAME..."
    aws ec2 create-key-pair --profile "$PROFILE" --region "$REGION" \
        --key-name "$KEY_NAME" \
        --query "KeyMaterial" --output text > ~/.ssh/"${KEY_NAME}".pem
    chmod 400 ~/.ssh/"${KEY_NAME}".pem
    echo "      Saved private key to ~/.ssh/${KEY_NAME}.pem"
else
    echo "      Reusing existing key pair $KEY_NAME"
fi

# 6. Launch Template / Instance with User Data (Cloud-Init)
echo "[6/6] Launching Graviton EC2 instance ($INSTANCE_TYPE)..."

USER_DATA=$(cat <<'EOF'
#!/bin/bash
set -ex
apt-get update
apt-get install -y docker.io git python3-pip
systemctl enable --now docker

# Clone AeroGuardian and start container
cd /opt
git clone https://github.com/IshekKhal/AeroGuardian.git || true
cd AeroGuardian
docker build -t aeroguardian:latest .
docker run -d --name aeroguardian-hud --restart unless-stopped \
    -p 8000:8000 -p 14550:14550/udp aeroguardian:latest
EOF
)

ENCODED_USER_DATA=$(echo "$USER_DATA" | base64)

INSTANCE_ID=$(aws ec2 run-instances --profile "$PROFILE" --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --user-data "$ENCODED_USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=AeroGuardian-Graviton-Node},{Key=Project,Value=OpenCV-AI-Competition-2026}]" \
    --query "Instances[0].InstanceId" --output text)

echo "=== Graviton Instance Launched Successfully ==="
echo "Instance ID: $INSTANCE_ID"
echo "Waiting for public IP assignment..."

aws ec2 wait instance-running --profile "$PROFILE" --region "$REGION" --instance-ids "$INSTANCE_ID"
PUBLIC_IP=$(aws ec2 describe-instances --profile "$PROFILE" --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

echo "----------------------------------------------------------------------"
echo "Public IPv4:     $PUBLIC_IP"
echo "Tactical HUD:    http://$PUBLIC_IP:8000"
echo "MAVLink Socket:  udp://$PUBLIC_IP:14550"
echo "SSH Access:      ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
echo "----------------------------------------------------------------------"
echo "To terminate instance when evaluation ends:"
echo "aws ec2 terminate-instances --profile $PROFILE --region $REGION --instance-ids $INSTANCE_ID"
echo "----------------------------------------------------------------------"
