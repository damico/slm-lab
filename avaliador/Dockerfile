FROM maven:3.9.9-eclipse-temurin-17 AS build
WORKDIR /workspace
COPY avaliador/pom.xml avaliador/pom.xml
COPY laboratorio-*.md ./
COPY avaliador/src avaliador/src
RUN mvn -B -ntp -f avaliador/pom.xml -DskipTests package
FROM eclipse-temurin:17-jre
WORKDIR /app
COPY --from=build /workspace/avaliador/target/avaliador-1.0.0.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "/app/app.jar"]
