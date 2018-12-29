To build Python wheel:
```bash
docker run -it --rm --net=host -v /python/to/euler:/tmp/Euler centos:7 /tmp/Euler/tools/pip/build_wheel.sh --tf_version <tf_version>
```
