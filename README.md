## Some important remarks to this software


### Master Node
 - Zerotier has to be installed with `make zerotier-install` and connected with `make zerotier` (In order for the connection to work, you have to setup the network id `ZEROTIER_JOIN_ID` env var, see `.env`)
 - All containerized services have to be up - `docker compose up`
 - There WILL BE IMPLEMENTED some CLI to distribute tasks to the slave workers



### Worker Node
 - Fetch and run image ... `TODO`

