### Setting up remote worker

connect with SSH:
`ssh -R 27017:localhost:27017 -R 6379:localhost:6379 goal-admin@connect.goalsport.software -p 10667`



Install Docker:
```
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Clone git repo
```
git clone https://github.com/michal-glos-brq/LunarPitsResearchToolbox.git
```


Daemon
```
sudo usermod -aG docker $USER

sudo systemctl start docker
sudo systemctl enable docker
```



Zero-Tier
```
curl -s https://install.zerotier.com | sudo bash


sudo systemctl enable zerotier-one
sudo systemctl start zerotier-one

sudo zerotier-cli join 60ee7c034a42cc48
```