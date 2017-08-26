#!/bin/bash
aws ec2 disassociate-address --association-id eipassoc-6432db50
aws ec2 release-address --allocation-id eipalloc-4903317a
aws ec2 terminate-instances --instance-ids i-0fbfabff4a1947f71
aws ec2 wait instance-terminated --instance-ids i-0fbfabff4a1947f71
aws ec2 delete-security-group --group-id sg-b01c87c0
aws ec2 disassociate-route-table --association-id rtbassoc-c8c75db2
aws ec2 delete-route-table --route-table-id rtb-678a461c
aws ec2 detach-internet-gateway --internet-gateway-id igw-14404f72 --vpc-id vpc-4e082437
aws ec2 delete-internet-gateway --internet-gateway-id igw-14404f72
aws ec2 delete-subnet --subnet-id subnet-2f82cc75
aws ec2 delete-vpc --vpc-id vpc-4e082437
echo If you want to delete the key-pair, please do it manually.
