# Download

````
$ docker pull xt1800i/snapnews-rec
````

# Deployment

### Step 1: install docker
Please refer to  [this link](https://www.linode.com/docs/applications/containers/install-docker-ce-ubuntu-1804/)

### Step 2: pull docker image
````
$ docker pull xt1800i/snapnews-rec
````

### Step 3: run docker container

````
$ docker run -itd -p 5002:5002 --restart=always  xt1800i/snapnews-rec
````


