## Some important remarks to this software


### Master Node
 - Zerotier has to be installed with `make zerotier-install` and connected with `make zerotier` (In order for the connection to work, you have to setup the network id `ZEROTIER_JOIN_ID` env var, see `.env`)
 - All containerized services have to be up - `docker compose up`
 - There WILL BE IMPLEMENTED some CLI to distribute tasks to the slave workers



### Worker Node
 - Fetch and run image ... `TODO`


### Remotely shared content
 - http://sialstoma.cz
   - NETWORK_ID - Id of zerotier network
   - MASTER_IP - Ip of master node for access

 - Lunar model with 7.5 % scale from original TIF file is on google drive:
   - https://drive.google.com/drive/u/0/folders/179ddA9nRm_ilpkgnmAtAOJO5noHTQYcH
