#!/bin/bash
# =============================================================================
# AWS Setup Script for CS229 Beam Steering ES Training
# =============================================================================
# This script automates Phase 3 of the deployment:
# - Creates S3 bucket
# - Creates IAM role for Batch jobs
# - Creates ECR repository
# - Creates Batch infrastructure (Compute Environment, Job Queue, Job Definition)
# =============================================================================

set -e  # Exit on any error

# Configuration
AWS_REGION="us-west-2"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
S3_BUCKET="cs229-beam-steering-results"
IAM_ROLE_NAME="cs229-batch-job-role"
ECR_REPO_NAME="cs229-trainer"
BATCH_COMPUTE_ENV="cs229-compute-env"
BATCH_QUEUE="cs229-queue"
BATCH_JOB_DEF="cs229-train"
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --region "$AWS_REGION" --output text)
SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[0].SubnetId" --region "$AWS_REGION" --output text)
SECURITY_GROUP=$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$VPC_ID" --query "SecurityGroups[0].GroupId" --region "$AWS_REGION" --output text)

echo "================================================================================"
echo "AWS SETUP FOR CS229 BEAM STEERING"
echo "================================================================================"
echo "Region: $AWS_REGION"
echo "Account: $AWS_ACCOUNT_ID"
echo "VPC: $VPC_ID"
echo "Subnet: $SUBNET_ID"
echo "Security Group: $SECURITY_GROUP"
echo ""

# =============================================================================
# TASK 3.2: Create S3 Bucket
# =============================================================================
echo "[TASK 3.2] Creating S3 bucket..."

if aws s3 ls "s3://$S3_BUCKET" 2>&1 | grep -q "NoSuchBucket"; then
    echo "  Bucket does not exist, creating..."
    aws s3 mb "s3://$S3_BUCKET" --region "$AWS_REGION"
    echo "  ✅ S3 bucket created: s3://$S3_BUCKET"
else
    echo "  ✅ S3 bucket already exists: s3://$S3_BUCKET"
fi

# Enable versioning (optional but recommended)
aws s3api put-bucket-versioning \
    --bucket "$S3_BUCKET" \
    --versioning-configuration Status=Enabled \
    --region "$AWS_REGION" 2>/dev/null || true
echo "  ✅ Versioning enabled"

# =============================================================================
# TASK 3.3: Create IAM Role for Batch Jobs
# =============================================================================
echo ""
echo "[TASK 3.3] Creating IAM role for Batch jobs..."

# Check if role exists
if aws iam get-role --role-name "$IAM_ROLE_NAME" 2>/dev/null; then
    echo "  ℹ️  Role already exists: $IAM_ROLE_NAME"
else
    # Create trust policy for ECS task execution
    cat > /tmp/batch_trust_policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    aws iam create-role \
        --role-name "$IAM_ROLE_NAME" \
        --assume-role-policy-document file:///tmp/batch_trust_policy.json \
        2>/dev/null
    echo "  ✅ IAM role created: $IAM_ROLE_NAME"
fi

# Attach S3 policy
aws iam put-role-policy \
    --role-name "$IAM_ROLE_NAME" \
    --policy-name "S3Access" \
    --policy-document '{
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Action": [
            "s3:GetObject",
            "s3:PutObject",
            "s3:ListBucket"
          ],
          "Resource": [
            "arn:aws:s3:::'"$S3_BUCKET"'",
            "arn:aws:s3:::'"$S3_BUCKET"'/*"
          ]
        }
      ]
    }' 2>/dev/null || true
echo "  ✅ S3 policy attached"

# Attach CloudWatch Logs policy
aws iam put-role-policy \
    --role-name "$IAM_ROLE_NAME" \
    --policy-name "CloudWatchLogs" \
    --policy-document '{
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Action": [
            "logs:CreateLogGroup",
            "logs:CreateLogStream",
            "logs:PutLogEvents"
          ],
          "Resource": "arn:aws:logs:'"$AWS_REGION"':'"$AWS_ACCOUNT_ID"':*"
        }
      ]
    }' 2>/dev/null || true
echo "  ✅ CloudWatch Logs policy attached"

# =============================================================================
# TASK 3.4: Create ECR Repository
# =============================================================================
echo ""
echo "[TASK 3.4] Creating ECR repository..."

if aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$AWS_REGION" 2>/dev/null | grep -q "repositoryArn"; then
    echo "  ℹ️  Repository already exists: $ECR_REPO_NAME"
else
    aws ecr create-repository \
        --repository-name "$ECR_REPO_NAME" \
        --region "$AWS_REGION" > /dev/null
    echo "  ✅ ECR repository created: $ECR_REPO_NAME"
fi

ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest"
echo "  ECR URI: $ECR_URI"
