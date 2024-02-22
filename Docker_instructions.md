## Docker

(With many thanks to Ernesto Casblanca for his help creating the Dockerfile and these instructions!)

These instructions may require sudo access.

### Building

If you go the `Docker` route, you only need to install docker as a dependency.
Then you can build the image and run the container with the following commands:

```bash
docker build -t impact .
```

The first time it may take a while.

### Running

Then you can run the container with the following command:

```bash
docker run --rm -it --name impact impact
```

This will open a shell inside the container.
You can then run the examples with the following commands:

```bash
cd /impact/examples
cd <specific example>
make
```

#### Mounting a volume

Furthermore, it is possible to mount a volume to the container to add more examples and run them in the container.
For example, if you create a folder `my-examples` in your home directory, you can mount it to the container with the following command:

```bash
docker run --rm -it --name impact -v ~/my-examples:/impact/my-examples impact
```

And then you can run the examples with the following commands:

```bash
cd /impact/my-examples
cd <specific example>
make
```

All files will be shared between the container and your host machine, so you can find the results in the `my-examples` folder on your host machine.

> [!CAUTION]
> Once the container is stopped, all the files created inside the container are lost.
> If you want to keep the results, you need to copy them to your host machine or [mount a volume](#mounting-a-volume) to the container.

### Obtaining the results

The results are stored in the in the same folder the example is located in.
To copy them to your host machine, you can use use a [volume](#mounting-a-volume) or the `docker cp` command to copy them to your host machine.

```bash
docker cp impact:/impact/my-examples/<specific example>/<result file> .
# e.g.
# docker cp impact:/app/examples/ex_2Drobot-R-U/fig.png .
```


