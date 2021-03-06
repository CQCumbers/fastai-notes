# Parameters defaults
# The size of the root volume, in GB.
volume_size=128
# The name of the key file we'll use to log into the instance. create_vpc.sh sets it to aws-key-fast-ai
name=fast-ai
key_name=aws-key-$name
ami=ami-da05a4a0
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

add-apt-repository ppa:graphics-drivers/ppa -y
apt-get update
apt-get install -y --allow-unauthenticated nvidia-375 nvidia-settings nvidia-modprobe
apt-get install -y zsh
apt-get install -y stow
apt-get install -y python3-pip
echo 'installed apt packages'

cd /home/ubuntu
pip3 install --upgrade pip
pip3 install numpy Pillow pandas matplotlib requests
pip3 install jupyter
pip3 install tensorflow-gpu
pip3 install h5py pydot_ng keras
pip3 install http://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl
pip3 install torchvision
mkdir .jupyter
jupass=\`python3 -c 'from notebook.auth import passwd; print(passwd(passphrase="dl_course"))'\`
echo "c.NotebookApp.ip = '*'\nc.NotebookApp.open_browser = False\nc.NotebookApp.password = '\$jupass'" > /home/ubuntu/.jupyter/jupyter_notebook_config.py
echo 'installed python packages'

sudo -H -u ubuntu zsh -c 'git clone --recursive https://github.com/sorin-ionescu/prezto.git ./.zprezto'
sudo -H -u ubuntu zsh -c 'git clone https://github.com/CQCumbers/dotfiles.git'
sudo -H -u ubuntu zsh -c 'git clone https://github.com/CQCumbers/fastai-notes.git'
echo 'cloned repos'

cd dotfiles
sudo -H -u ubuntu zsh -c 'stow zsh'
sudo -H -u ubuntu zsh -c 'stow vim'
cd /home/ubuntu
sudo -H -u ubuntu zsh -c 'vim -E -c PlugClean -c PlugUpdate -c q'
echo 'loaded dotfiles'

dpkg --configure -a
wget "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb"
dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
apt-get update
apt-get install -y --allow-unauthenticated cuda-8-0
sudo -H -u ubuntu zsh -c 'nvidia-smi'
wget "http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn6_6.0.21-1+cuda8.0_amd64.deb"
dpkg -i libcudnn6_6.0.21-1+cuda8.0_amd64.deb
apt-get install -f -y --allow-unauthenticated
modprobe nvidia
echo 'installed cuda'

echo 'alias python="python3"' >> /home/ubuntu/.zshrc
echo 'alias pip="pip3"' >> /home/ubuntu/.zshrc
echo 'export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/cuda-8.0/lib64' >> /home/ubuntu/.zshrc
echo 'export PATH=\${PATH}:/usr/local/cuda-8.0/bin' >> /home/ubuntu/.zshrc
chown ubuntu:ubuntu /home/ubuntu/
chsh ubuntu -s /bin/zsh
echo 'setup user environment'
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
        "DeleteOnTermination": true, 
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
