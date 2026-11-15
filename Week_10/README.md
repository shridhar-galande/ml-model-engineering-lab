# Kafka Streaming

This folder contains a small, classroom-friendly Kafka streaming demo that shows how a producer, broker, and consumer work together.

## Files
- `docker-compose.yml` — starts Kafka and the Kafka UI locally
- `requirements.txt` — Python dependencies needed for the demo
- `producer.py` — sends simulated transaction events to Kafka
- `consumer.py` — reads events and performs simple quality checks
- `kafka_streaming_lab.ipynb` — step-by-step notebook for teaching and live demos
- `lab_guide.md` — instructor-facing walkthrough

## Quick start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the stack:
   ```bash
   docker compose up -d
   ```
3. Open the notebook and run the cells in order.
4. When finished:
   ```bash
   docker compose down -v
   ```

## Notes
- Kafka UI is available at http://localhost:8080
- The default broker address is `localhost:9092`
