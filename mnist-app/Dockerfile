FROM adoptopenjdk/openjdk11:alpine-jre
COPY ./mnist-app.jar /mnist-app.jar
ENTRYPOINT ["java","-jar","-XX:+UseSerialGC", "/mnist-app.jar"]