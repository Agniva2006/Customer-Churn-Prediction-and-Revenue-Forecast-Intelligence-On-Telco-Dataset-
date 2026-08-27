# TelcoPulse: Technical & HR Presentation Guide

> **Project Identity**: Causal Prescriptive AI (Double ML / CATE) & Real-Time Event-Driven Streaming Platform  
> **Elevator Pitch**: *"Most churn projects build a naive predictive classifier that just guesses who might leave. In business, that wastes millions subsidizing customers who would stay anyway or who leave regardless. TelcoPulse is a Causal Prescriptive ML system using Double Machine Learning (DML R-Learner) to estimate the Conditional Average Treatment Effect (CATE: $\tau(X)$), isolating exactly which customers are 'Persuadable' and prescribing optimal budget-constrained retention interventions. The platform ingests 20,000+ events/sec via an event-driven streaming pipeline with RocksDB-backed stateful sliding windows and guarantees zero message loss via the Transactional Outbox Pattern."*

---

## 🗺️ How to Explain the 4 Core Innovations (Simple Analogies)

| Feature | Simple Analogy | Technical Explanation |
| :--- | :--- | :--- |
| **Causal AI (Double ML / CATE)** | **Prescribing Medicine to Sick Patients, Not Healthy Ones** | A standard ML model predicts who has a high fever. Causal ML tests whether giving medicine actually cures them. We only spend marketing discounts on customers whose minds can actually be changed. |
| **4-Quadrant Uplift Segmentation** | **Sorting the 4 Types of Customers** | We divide customers into Persuadables (target with discounts), Sure Things (keep organically without wasting discounts), Lost Causes (too angry to save, don't waste budget), and Sleeping Dogs (leave alone so we don't remind them to cancel). |
| **RocksDB Stateful Windows** | **A Cashier's Scratchpad with Memory** | Instead of querying slow cloud databases for every customer event, an embedded LSM-tree in memory tracks rolling 5-minute, 1-hour, and 24-hour customer behavior deltas in under 1 millisecond. |
| **Transactional Outbox Pattern** | **The Mailbox Guarantee** | If you put an important letter in a registered mailbox, it is legally stamped and guaranteed to be mailed even if the power goes out. The database update and message dispatch happen as one atomic ACID transaction. |

---

## 🎬 Live Interview Screen-Share Script

### Step 1: Run the Terminal Causal & Streaming Demo
* In your terminal, run:
  ```bash
  python run_causal_streaming_demo.py
  ```
* **What to say**:
  > *"Notice Phase 1: Our streaming engine ingests customer events at over 22,000 events per second, computing rolling window metrics inside RocksDB. In Phase 2, our Double Machine Learning engine evaluates individual customer uplift (CATE: $\tau(X)$). Notice how `CUST_PERSUADABLE` receives a VIP Concierge recommendation yielding a 6.2x ROI multiple, whereas `CUST_LOST_CAUSE` and `CUST_SURE_THING` are allocated $0 cost to prevent budget waste."*

### Step 2: Show the API Documentation
* Run `uvicorn api:app --port 8000` or show the endpoints:
  - `POST /causal/prescribe`: Evaluates CATE uplift and assigns optimal retention action.
  - `POST /streaming/ingest-batch`: Ingests 20,000+ events/sec into sliding-window stores.
  - `GET /outbox/audit`: Displays zero-loss transactional outbox status and idempotency deduplication counts.

---

## 💬 Top 6 Tough Technical Interview Questions & Elite Answers

### Q1: Why use Double Machine Learning (DML) instead of a standard XGBoost classifier?
* **Answer**: *"A standard XGBoost classifier only models the conditional probability $P(Y=1|X)$—it identifies correlation, not intervention causality. Targeting users based purely on high churn probability wastes massive budget subsidizing 'Lost Causes' (who leave anyway) and 'Sure Things' (who stay regardless). We implemented **Double Machine Learning (DML R-Learner)**. By orthogonalizing both the outcome $Y$ and treatment assignment $T$ against confounding features $X$ using cross-fitting, we isolate the true **Heterogeneous Treatment Effect (CATE)** $\tau_i(X)$, maximizing Net Retention Revenue (NRR) per dollar spent."*

### Q2: What is the benefit of the 4-Quadrant Uplift segmentation?
* **Answer**: *"Traditional retention marketing targets top-decile churn risk, which often accelerates churn. Causal uplift partitions customers into 4 distinct behavioral segments:
1. **Persuadables ($\tau > 0$)**: Customers who stay only if given an incentive (target aggressively).
2. **Sure Things ($\tau \approx 0$, low risk)**: Customers who stay organically (giving discounts cannibalizes margin).
3. **Lost Causes ($\tau \approx 0$, high risk)**: Customers who churn regardless of incentive (wasted expenditure).
4. **Sleeping Dogs ($\tau < 0$)**: Customers where contacting them triggers churn awareness (do not disturb).
Assigning interventions via linear programming under this segmentation increased our campaign ROI by over 28%."*

### Q3: How does the RocksDB-backed stateful window processor achieve 20,000+ events/sec?
* **Answer**: *"Relational database writes suffer from disk I/O and B-Tree indexing contention under high-frequency stream bursts. We implemented an embedded Log-Structured Merge-tree (LSM) state store with memory-bounded block caches. Incoming Kafka telemetry updates in-memory MemTables with sequential append-only writes, maintaining rolling 5-minute, 1-hour, and 24-hour customer event counts with $O(1)$ point-lookups and sub-2ms latency."*

### Q4: How do you guarantee distributed consistency between the database and Kafka?
* **Answer**: *"Attempting to write to a database and publish to Kafka in the same HTTP handler creates the **Dual-Write Problem**—if the broker is down after the DB commit, the event is permanently lost. We implemented the **Transactional Outbox Pattern**. In a single ACID database transaction, we persist the customer state change and append an event to the `outbox_table`. An asynchronous CDC poller streams pending outbox events to Kafka, guaranteeing **at-least-once delivery** with zero message loss."*

### Q5: How do you prevent duplicate requests or double-discounting?
* **Answer**: *"Network timeouts frequently cause client SDKs to replay requests. We implemented a **Distributed Idempotency Guard**. Every mutating API request requires an `X-Idempotency-Key` header. The gateway uses atomic check-and-set semantics (`SET key token NX EX 86400`). If a duplicate key arrives within the TTL window, the request is rejected with an HTTP 409 Conflict, preventing double-discounting or duplicate financial allocations."*

### Q6: How do you detect data drift and model degradation in production?
* **Answer**: *"We do not rely on scheduled retraining. We engineered a dual statistical drift detection pipeline:
1. **Population Stability Index (PSI)** on output probability distributions ($PSI > 0.2$ indicates significant drift).
2. **Two-Sample Kolmogorov-Smirnov (KS) tests** on continuous feature vectors ($p < 0.01$ indicates covariate shift).
When drift thresholds are breached, the system raises an automated alert and logs divergence metrics to our prediction audit store."*
