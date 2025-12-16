#!/bin/bash
set -e

AWS_REGION="us-west-2"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
COMPUTE_ENV_NAME="cs229-beam-steering-compute-env"

# Get current network config
DEFAULT_VPC=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text)
DEFAULT_SUBNET=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$DEFAULT_VPC" --query 'Subnets[0].SubnetId' --output text)
DEFAULT_SG=$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$DEFAULT_VPC" "Name=group-name,Values=default" --query 'SecurityGroups[0].GroupId' --output text)
INSTANCE_PROFILE="arn:aws:iam::${AWS_ACCOUNT_ID}:instance-profile/cs229-beam-steering-ec2-instance-profile"

echo "Updating Compute Environment for SPEED..."
echo "  Account: $AWS_ACCOUNT_ID"
echo "  Region: $AWS_REGION"
echo "  VPC: $DEFAULT_VPC"
echo "  Subnet: $DEFAULT_SUBNET"
echo "  Security Group: $DEFAULT_SG"

# Update compute resources: Enable minvCpus=1, remove Spot (no bidPercentage)
aws batch update-compute-environment \
    --compute-environment "$COMPUTE_ENV_NAME" \
    --compute-resources \
        type=EC2,\
minvCpus=1,\
maxvCpus=96,\
desiredvCpus=1,\
instanceTypes=optimal,\
subnets="$DEFAULT_SUBNET",\
securityGroupIds="$DEFAULT_SG",\
instanceRole="$INSTANCE_PROFILE" \
    --region "$AWS_REGION"

echo "✅ Compute Environment updated for SPEED"
echo "  - Removed Spot instances (bidPercentage deleted)"
echo "  - Set minvCpus: 1 (warm instance always ready)"
echo "  - Set desiredvCpus: 1"
echo "  - maxvCpus: 96 (can scale up when needed)"
echo "  - Using On-Demand instances only"
