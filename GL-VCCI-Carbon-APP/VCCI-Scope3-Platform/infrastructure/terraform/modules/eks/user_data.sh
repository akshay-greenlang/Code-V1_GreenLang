#!/bin/bash
set -o xtrace

# Bootstrap EKS worker node
/etc/eks/bootstrap.sh ${cluster_name} \
  --b64-cluster-ca '${cluster_ca}' \
  --apiserver-endpoint '${cluster_endpoint}' \
  --kubelet-extra-args '--node-labels=node.kubernetes.io/lifecycle=normal'

# Install additional tools
yum install -y amazon-ssm-agent
systemctl enable amazon-ssm-agent
systemctl start amazon-ssm-agent

# Configure logging
cat > /etc/docker/daemon.json <<EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF

systemctl restart docker
