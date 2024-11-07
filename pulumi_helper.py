import pulumi
import pulumi_aws as aws
from pulumi import FileArchive, AssetArchive, Output


def get_latest_linux2(owner = "amazon"):
    """
    This function return the latest linux 2 ami you want to use for EC2, the default value is amazon
    Args:
        owner: the owner of the latest linux 2 ami you want to pull

    Returns:
        the latest linux 2 ami object from selected owner.
    """
    ami = aws.ec2.get_ami(
    most_recent=True,
    owners=[owner],
    filters=[{"name": "name", "values": ["amzn2-ami-hvm-*-x86_64-gp2"]}],
)
    return ami

def get_default_subnet():

    """
    This function return the default subnet from default vpn from the first return value of availability zone
    Returns:
    default_subnet: default subnet from default vpn from the first return value of availability zone
    """
    default_vpc = aws.ec2.get_vpc(default=True)
    availability_zones = aws.get_availability_zones()

    # Select the first availability zone (you can choose any)
    selected_az = availability_zones.names[0]  # availability_zones.names[0]

    # Get the default subnet in the selected availability zone
    default_subnet = aws.ec2.get_subnet(
        filters=[
            aws.ec2.GetSubnetFilterArgs(
                name="vpc-id",
                values=[default_vpc.id],
            ),
            aws.ec2.GetSubnetFilterArgs(
                name="availability-zone",
                values=[selected_az],
            ),
            aws.ec2.GetSubnetFilterArgs(
                name="default-for-az",
                values=["true"],
            ),
        ],
    )

    return default_subnet

def get_ec2_assumerole():

    """
    The function return an assume role policy that allow EC2 to assume the iam role created later that uses this policy

    statements: The policy document can contain multiple statements,
                each specifying a different aspect of permissions.
                Here, we are defining a single statement, which is to allow the ec2 (principals) to assume role

    aws.iam.GetPolicyDocumentStatementArgs(): This is defining one statement within the policy document.

    effect="Allow": This part specifies that the effect of the policy is to "Allow" the action defined.
                    This means we are allowing the entity described in Principal to take the action.

    principals: The principal is the entity that is allowed to assume the role. In this case, it is EC2.

    aws.iam.GetPolicyDocumentStatementPrincipalArgs(...): This is specifying the type of principal and its identifier.
                                                          type="Service": This means the principal is a service
                                                                (as opposed to a user or another type of entity).
                                                          identifiers=["ec2.amazonaws.com"]: The specific service
                                                                                             allowed to assume the role
                                                                                             is EC2. This allows
                                                                                             EC2 instances to assume
                                                                                             this IAM role.

    actions=["sts:AssumeRole"]: This specifies what action the principal is allowed to perform. "sts:AssumeRole" means
                                that the principal (EC2) is allowed to assume this IAM role via
                                AWS Security Token Service (STS). This is crucial for EC2 instances to be able to take
                                on the permissions assigned to the role.
    Returns:
    assume_role_policy: assume role policy that allows EC2 to assume the IAM role created later
    """

    assume_role_policy = aws.iam.get_policy_document(
        statements=[aws.iam.GetPolicyDocumentStatementArgs(
            effect="Allow",
            principals=[aws.iam.GetPolicyDocumentStatementPrincipalArgs(
                type="Service",
                identifiers=["ec2.amazonaws.com"],
            )],
            actions=["sts:AssumeRole"],
        )],
    )

    return assume_role_policy

def create_iam_with_asumme(name = "ec2-role", pre_role = None):
    """
    This function actually creates an return the IAM role with specified name and assume policy
    Args:
        name: name you want to give this IAM role during this IAS environment
        pre_role: previous defined assume role policy

    Returns:
    role: the created IAM role with pre_defined assume role policy
    """
    if not pre_role:
        raise ValueError
    role = aws.iam.Role(name, assume_role_policy=pre_role.json)
    return role

def vpc_security_group(name = "flask-app-sg", able_port = 5000, ssh_ip = "0.0.0.0/0"):

    """
    The function return a defined security group that enable port 5000 and SSH connection
    By default, it will allow any http access to port 5000 and SSH access from any where unless specify ssh_ip
    Args:
        name: name of the security_group, default is flask-app-sg
        able_port: port number you want to enable to access from anythwere, default value is 5000
        ssh_ip: defined IP address allowed to SSH access, default value is 0.0.0.0/0 (anywhere)

    Returns:
    security_group
    """

    security_group = aws.ec2.SecurityGroup(
        "flask-app-sg",
        description="Enable port 5000 and SSH",
        ingress=[
            # SSH access from your IP (recommended)
            aws.ec2.SecurityGroupIngressArgs(
                protocol="tcp",
                from_port=22,
                to_port=22,
                cidr_blocks=[ssh_ip],  # Use your IP if provided, else 0.0.0.0/0
                description="SSH access from my IP or anywhere"
            ),
            # HTTP access from anywhere
            aws.ec2.SecurityGroupIngressArgs(
                protocol="tcp",
                from_port=80,
                to_port=80,
                cidr_blocks=["0.0.0.0/0"],
                description="HTTP access from anywhere"
            ),
            # Custom TCP port (e.g., 5000) access from anywhere
            aws.ec2.SecurityGroupIngressArgs(
                protocol="tcp",
                from_port=able_port,
                to_port=able_port,
                cidr_blocks=["0.0.0.0/0"],
                description="Custom TCP port 5000 access from anywhere"
            ),
        ],
        egress=[
            # Allow all outbound traffic
            aws.ec2.SecurityGroupEgressArgs(
                protocol="-1",
                from_port=0,
                to_port=0,
                cidr_blocks=["0.0.0.0/0"],
                description="Allow all outbound traffic"
            ),
        ],
        tags={
            "Name": "FlaskAppSecurityGroup"
        }
    )

    return security_group

def create_upload_s3(name_bucket = "app-bucket", file_path = "app/", name_archive = "app-archive", name_in_s3 = "app.zip"):

    """

    Args:
        name_bucket: name of the created S3 bucket in IAS
        file_path: path of the file or directory needed to upload to s3, will be uploaded in .zip
        name_archive: name of the s3 bucket object in IAS
        name_in_s3: name of the uploaded file in created S3 bucket

    Returns:
    app_bucket, app_archive_object: created S3 bucket, created S3 bucket object (file uploaded)
    """

    app_bucket = aws.s3.Bucket(name_bucket)

    # Archive the app directory
    app_archive = FileArchive(file_path)

    # Upload the app archive to S3
    app_archive_object = aws.s3.BucketObject(
        name_archive,
        bucket=app_bucket.id,
        key=name_in_s3,
        source=app_archive,
        acl="private",
    )

    return app_bucket, app_archive_object


def create_user_data_script(bucket_name, object_key):

    """
    This function returns the user_data_script that will be used in creating EC2 instance. Basically it update the
    packages inside the created EC2 instance, and install docker, docker-compose for running the program.
    It also takes two arguments that used to retrieve required documents from s3 to the instance
    Args:
        bucket_name: name of the bucket created that store the required file
        object_key: id of the s3 bucket object created related to the required file

    Returns:

    """

    return f"""#!/bin/bash
# Update the system

exec > /var/log/user_data.log 2>&1
set -x
sudo yum update -y

# Install Docker
sudo amazon-linux-extras install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep tag_name | cut -d '"' -f 4)/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install AWS CLI and unzip
sudo yum install -y awscli unzip

# Switch to ec2-user's home directory
cd /home/ec2-user

# Download the application archive from S3
aws s3 cp s3://{bucket_name}/{object_key} app.zip

# Unzip the archive
unzip app.zip -d app

# Change ownership of the app folder
sudo chown -R ec2-user:ec2-user app

# Move into the app directory
cd app

# Run Docker Compose
docker-compose build
docker-compose up -d
"""

def processed_user_data_script(app_bucket, app_archive_object, create_user_data_script):

    """
    This function processes the user_date_script created by create_user_data_script() and make sure that it will only
    generate the user_date_script after the app_bucket and app_archive_object are fully ready to avoid dependency error
    Args:
        app_bucket: name of the bucket created that store the required file
        app_archive_object: id of the s3 bucket object created related to the required file
        create_user_data_script: function defined above to create use_data_script

    Returns:

    """

    user_data_script = pulumi.Output.all(app_bucket.bucket, app_archive_object.id).apply(
        lambda args: create_user_data_script(args[0], args[1])
    )

    return user_data_script