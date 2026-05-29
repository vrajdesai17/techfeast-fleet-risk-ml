# Fleet Risk ML — IIT Bombay TECHFEST 2022 Winner 🏆

A machine learning system for **privacy-preserving fleet risk scoring** built using **TensorFlow Federated**. Predicts driver risk, route risk, and vehicle risk from distributed telematics data without centralizing sensitive trip information.

> 🥇 **1st Place — IIT Bombay TECHFEST 2022 Fleet Risk ML Contest**
> Competed against 100 global teams. Led a 3-person engineering team.

---

## Problem

Fleet operators want to identify high-risk drivers, routes, and vehicles to reduce accidents and insurance costs. But the telematics data needed for risk scoring — driving patterns, locations, behaviors — is highly sensitive. Centralizing it creates privacy and regulatory risk.

**Solution:** Federated learning. Train models locally on each vehicle's data, aggregate only model updates (not raw data) into a global risk-scoring system.

---

## Three Risk Models

The system trains three complementary models, each evaluating a different dimension of fleet risk:

| Model | File | Predicts |
|---|---|---|
| **Driver Risk** | `drivermodel.py` | Risk score based on driving behavior (acceleration, braking, speed patterns) |
| **Route Risk** | `routemodel.py` | Risk score based on route characteristics (traffic, road quality, weather exposure) |
| **Vehicle Risk** | `vehiclemodel.py` | Risk score based on vehicle telemetry (engine health, maintenance signals) |

Each model is trained via **TensorFlow Federated**, so raw telematics data never leaves the source vehicle.

---

## Why Federated Learning?

Traditional ML for fleet risk requires centralizing data from every vehicle into one server — creating:
- **Privacy risk** — driving patterns can identify individuals
- **Regulatory exposure** — GDPR, regional data sovereignty laws
- **Bandwidth cost** — telematics data is high-volume
- **Single point of failure** — central server breach exposes all fleet data

Federated learning solves all four. Each vehicle (client) trains locally; only encrypted model weights are aggregated. The global model improves without ever seeing raw trips.

---

## Architecture

<img width="599" height="453" alt="image" src="https://github.com/user-attachments/assets/81eacacb-3f6c-4f05-b38e-b53ec6c4c097" />

---

## Tech Stack

| Component | Technology |
|---|---|
| ML Framework | TensorFlow + TensorFlow Federated |
| Language | Python 3.8+ |
| Visualization | Tableau |
| Data | Simulated telematics dataset (contest provided) |

---

## Repository Structure

techfeast-fleet-risk-ml/
- DSP1/                # Data simulation / preprocessing pipeline 1
- DSP2/                # Data simulation / preprocessing pipeline 2
- drivermodel.py       # Federated driver-risk model
- routemodel.py        # Federated route-risk model
- vehiclemodel.py      # Federated vehicle-risk model
- README.md

---

## Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- TensorFlow Federated

### Install
```bash
pip install tensorflow tensorflow-federated pandas numpy scikit-learn
```

### Run a model
```bash
python drivermodel.py
python routemodel.py
python vehiclemodel.py
```

---

## Results

- **1st place** out of 100 global teams at IIT Bombay TECHFEST 2022
- Successfully demonstrated federated training across simulated multi-vehicle clients
- Generated Tableau dashboard surfacing high-risk routes and driver patterns to fleet operators
- Preserved data privacy throughout — no raw telematics centralized

---

## Team

Led a 3-person engineering team through:
- Federated learning architecture design
- Model training and tuning across three risk dimensions
- Dashboard design and final presentation

---

## License

MIT — see [LICENSE](LICENSE)

