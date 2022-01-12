docker build --tag=europe-central2-docker.pkg.dev/leartnouse/torch/serve-test .
docker stop local_test
docker rm local_test
docker run -d -p 8080:8080 -p 8081:8081 -p 8082:8082 --name=local_test europe-central2-docker.pkg.dev/leartnouse/torch/serve-test