# LearnLib JVM gateway

This directory contains a code to use [LearnLib](https://learnlib.de/) and the learning algorithm modified by us from python using [Py4J](https://www.py4j.org/). The code under this directory is originally from https://github.com/mtf90/learnlib-py4j-example, which is distributed under Apache-2.0 license.

## Usage

```sh
mvn package
java -jar target/learnlib-py4j-example-1.0-SNAPSHOT.jar
```

If your machine has enough RAM, please consider adding `-XX:+AggressiveHeap`, i.e., run `java -XX:+AggressiveHeap -jar target/learnlib-py4j-example-1.0-SNAPSHOT.jar`.

## Notable file

- `src/main/java/org/group_mmm/StrongBlueFringeRPNIMealy.java`: This is the file implementing our modified RPNI algorithm, which uses `min_depth`. We note that the part to adaptively change `min_depth` is implemented in python.
