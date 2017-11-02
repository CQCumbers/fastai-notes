# Parameters defaults
# The size of the root volume, in GB.
volume_size=128
# The name of the key file we'll use to log into the instance. create_vpc.sh sets it to aws-key-fast-ai
name=fast-ai
key_name=aws-key-$name
ami=ami-37bb714d
subnetId=subnet-390ce364
securityGroupId=sg-287d205a
# Type of instance to launch
ec2spotter_instance_type=p2.xlarge
# In USD, the maximum price we are willing to pay.
bid_price=1.0

# Read the input args
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --ami)
    ami="$2"
    shift # pass argument
    ;;
    --subnetId)
    subnetId="$2"
    shift # pass argument
    ;;
    --securityGroupId)
    securityGroupId="$2"
    shift # pass argument
    ;;
	--volume_size)
	volume_size="$2"
	shift # pass argument
	;;
	--key_name)
	key_name="$2"
	shift # pass argument
	;;
	--ec2spotter_instance_type)
	ec2spotter_instance_type="$2"
	shift # pass argument
	;;
	--bid_price)
	bid_price="$2"
	shift # pass argument
	;;
    *)
            # unknown option
    ;;
esac
shift # pass argument or value
done

# Create a startup script to run on instance boot
cat >user-data.tmp <<EOF
#!/bin/sh
chsh ubuntu -s /bin/nologin
echo AWSAccessKeyId=$AWS_ACCESS_KEY > /root/.aws.creds
echo AWSSecretKey=$AWS_SECRET_KEY >> /root/.aws.creds

apt-get install -y zsh
apt-get install -y stow
pip install awscli

cd /home/ubuntu
rm ./.zshrc
rm ./src
git clone --recursive https://github.com/sorin-ionescu/prezto.git ./.zprezto
git clone https://github.com/CQCumbers/dotfiles.git
git clone https://github.com/CQCumbers/fastai-notes.git
echo 'cloned repos'

cd dotfiles
sudo -H -u ubuntu zsh -c 'stow zsh'
sudo -H -u ubuntu zsh -c 'stow vim'
cd /home/ubuntu
sudo -H -u ubuntu zsh -c 'vim -E -c PlugClean -c PlugUpdate -c q'
echo 'loaded dotfiles'

wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
bash Anaconda3-4.2.0-Linux-x86_64.sh -b -p ./anaconda
rm Anaconda3-4.2.0-Linux-x86_64.sh
echo 'export PATH="/home/ubuntu/anaconda/bin:$PATH"' >> ./.zshrc 
sudo -H -u ubuntu zsh -c 'source ./.zshrc'
sudo -H -u ubuntu zsh -c 'conda update conda'
sudo -H -u ubuntu zsh -c 'conda create -n py36 python=3.6 anaconda'
sudo -H -u ubuntu zsh -c 'source activate py36'
sudo -H -u ubuntu zsh -c 'conda install theano pygpu'
sudo -H -u ubuntu zsh -c 'pip install keras'
chown -v -R ubuntu:ubuntu /home/ubuntu/
chsh ubuntu -s /bin/zsh
echo 'installed deep learning libraries'
EOF

userData=$(base64 user-data.tmp | tr -d '\n');

# Create a config file to launch the instance.
cat >specs.tmp <<EOF 
{
  "ImageId" : "$ami",
  "InstanceType": "$ec2spotter_instance_type",
  "KeyName" : "$key_name",
  "EbsOptimized": true,
  "BlockDeviceMappings": [
    {
      "DeviceName": "/dev/sda1",
      "Ebs": {
        "DeleteOnTermination": false, 
        "VolumeType": "gp2",
        "VolumeSize": $volume_size 
      }
    }
  ],
  "NetworkInterfaces": [
      {
        "DeviceIndex": 0,
        "SubnetId": "${subnetId}",
        "Groups": [ "${securityGroupId}" ],
        "AssociatePublicIpAddress": true
      }
  ],
  "UserData": "${userData}"
}
EOF

# Request the spot instance
export requestId=`aws ec2 request-spot-instances --launch-specification file://specs.tmp --spot-price $bid_price --output="text" --query="SpotInstanceRequests[*].SpotInstanceRequestId"`

echo Waiting for spot request to be fulfilled...
aws ec2 wait spot-instance-request-fulfilled --spot-instance-request-ids $requestId  

# Get the instance id
export instanceId=`aws ec2 describe-spot-instance-requests --spot-instance-request-ids $requestId --query="SpotInstanceRequests[*].InstanceId" --output="text"`

echo Waiting for spot instance to start up...
aws ec2 wait instance-running --instance-ids $instanceId

echo Spot instance ID: $instanceId 

# Change the instance name
aws ec2 create-tags --resources $instanceId --tags --tags Key=Name,Value=$name-gpu-machine

# Get the instance IP
export instanceIp=`aws ec2 describe-instances --instance-ids $instanceId --filter Name=instance-state-name,Values=running --query "Reservations[*].Instances[*].PublicIpAddress" --output=text`

echo Spot Instance IP: $instanceIp

# Clean up
rm specs.tmp
rm user-data.tmp
