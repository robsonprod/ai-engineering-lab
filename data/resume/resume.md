# Robson Azevedo — Senior Software Engineer · AI Engineering

> Public resume content, anonymized for this lab: employer and client names are replaced by
> a short description of the organization.

Location: Fortaleza, Brazil. Open to remote and hybrid roles.
Languages: Portuguese (native), English (professional), Spanish (basic).

## Summary

Senior Software Engineer focused on AI engineering: LLM applications, spec-driven development
and agentic workflows, backed by 10+ years of Java, AWS and distributed systems for regulated,
high-availability platforms.

Day-to-day is AI-first: specs before code with OpenSpec, agentic development with Claude Code
and MCP, and LLM features with LangChain.

## Areas of focus

- **AI-Native Engineering:** Spec-Driven Development (SDD) with OpenSpec; agentic coding with
  Claude Code, custom skills and MCP servers, with humans reviewing every step.
- **LLM Applications:** retrieval-augmented generation (RAG), tool calling, structured outputs and
  guardrails, built with LangChain and integrated into existing back-ends and APIs.
- **Back-end & Distributed Systems:** microservices with DDD, resilience patterns, idempotency and
  event-driven integration with Kafka, in Java and Kotlin on Spring Boot.
- **Cloud, DevOps & Front-end:** AWS infrastructure as code with Terraform, serverless and
  Kubernetes, zero-secret CI/CD with OIDC, Angular and Next.js front-ends.

## Experience

### Senior Software Engineer — technology consultancy (Jan 2023 – present)
Client: a large Brazilian bank.
- Applies AI-assisted engineering in daily delivery: spec-driven development, agentic coding and
  AI-assisted code review.
- Designs and builds highly scalable, highly available microservices with Java 17/21, Spring Boot,
  Spring Cloud and Kotlin, applying DDD and resilience patterns on critical financial platforms.
- Builds Angular micro-frontends and BFF layers in TypeScript/Next.js.
- Designs event-driven serverless solutions with AWS Lambda, SQS/SNS, DynamoDB and S3.
- Operates and evolves cloud-native infrastructure on AWS (EKS, ECS, RDS Aurora) with Terraform.
- Architects distributed systems with Apache Kafka and Cassandra: topologies, partitioning and
  retention policies.

### Senior Software Engineer — software R&D institute (Mar 2021 – Jan 2023)
Client: a global computer hardware manufacturer.
- Designed microservices and micro-frontend architectures with Java 11, Spring Boot,
  Spring Security, Spring Cloud and Angular.
- Built data pipelines and event-driven communication with Kafka, Oracle and QueryDSL.
- Deployed and managed applications on Pivotal Cloud Foundry running on Microsoft Azure.
- Led code reviews and technical mentoring across agile squads.

### Full Stack Developer — public-sector agency (Feb 2020 – Mar 2021)
- Maintained and evolved production RESTful systems with Java, Spring MVC, JSF, Hibernate and
  PostgreSQL.

### Full Stack Developer — software house (Oct 2019 – Feb 2020)
Client: a healthcare provider.
- Delivered features and supported production systems with Java, Spring Boot, Angular, Oracle and
  Hibernate.

### Full Stack Developer — logistics software company (Jan 2018 – Oct 2019)
- Evolved route-optimization systems with Java, Spring, JPA, PostgreSQL and SOAP/REST services.

### Web Developer — web software house (Jan 2015 – Jul 2016)
- Built systems with Java 7, Java EE and EJB3, accessing PostgreSQL with plain JDBC and
  hand-written SQL, no ORM.
- Front-ends with PrimeFaces and JavaScript; reporting with JasperReports.

## Skills

- AI Engineering: LangChain, RAG, OpenSpec / SDD, Claude Code, MCP, Agents, Prompt engineering
- Languages: Java (7 to 25), Kotlin, TypeScript, JavaScript, Python, SQL
- Back-end: Spring Boot, Spring Cloud, Spring Security, WebFlux, Hibernate/JPA, JDBC,
  Java EE / EJB3, REST, SOAP
- Cloud & IaC: EKS, ECS, Lambda, DynamoDB, S3, RDS Aurora, SQS/SNS, IAM, Terraform, Docker,
  Kubernetes
- Messaging & Data: Apache Kafka, Cassandra, DynamoDB, PostgreSQL, Oracle, MySQL, Redis
- Front-end & BFF: Angular, Micro-frontends, Next.js, BFF
- Architecture: Microservices, Event-driven, Serverless, DDD, Resilience, OAuth2/JWT, CI/CD

## Projects

### balance-consistency-service (completed)
Kafka-to-DynamoDB service that keeps account balances correct under duplicate, out-of-order and
failing messages, built for a public banking code challenge.
- Idempotency and ordering without locks: conditional TransactWriteItems (balance snapshot +
  ledger entry, all or nothing).
- DynamoDB single-table design with a GSI.
- Retry with backoff, dead-letter topic and a Resilience4j circuit breaker.
- Hexagonal architecture enforced by ArchUnit; 90% coverage gate plus end-to-end tests on real
  Kafka and DynamoDB; load test with Prometheus/Grafana; CI with build, tests, Docker and CodeQL.

### Portfolio on AWS (in progress)
Pre-rendered Angular app served from a private S3 bucket through CloudFront.
- Built with Claude Code plus human review of every change.
- Terraform with remote state; GitHub Actions to AWS via OIDC; strict Content-Security-Policy.

### bank-resilience-lab (in progress)
Distributed-systems lab on a simplified banking domain (Kotlin, Java 25, Spring Boot, PostgreSQL,
Testcontainers).
- Race conditions: ledger with 6 comparable concurrency strategies.
- Virtual threads vs platform threads benchmark.
- One decision record per step: problem, solution, trade-offs.

### ai-engineering-lab (in progress)
This lab, in Python: LLM fundamentals, RAG with citations, evals, a LangGraph agent with
human-in-the-loop, guardrails and observability.

## Education

- MBA in Software Engineering — Agile Processes (2017 – 2018)
- B.Sc. in Information Systems (2013 – 2016)
- Java Training (2017)
- Information Security Analyst Jr certification
