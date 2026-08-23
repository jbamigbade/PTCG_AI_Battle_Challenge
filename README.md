# Pokémon TCG AI Battle Challenge

> **Portfolio Project:** End-to-end AI/ML engineering system for Pokémon TCG battle-agent development, combining supervised learning, PyTorch-based PPO reinforcement learning, battle simulation, explainability, robustness testing, tournament benchmarking, deployment engineering, and official-engine validation.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-PPO%20%2F%20RL-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-orange)
![Status](https://img.shields.io/badge/Project-Complete-brightgreen)
![Validation](https://img.shields.io/badge/Official%20Decisions-127%2F127%20Accepted-brightgreen)

## End-to-End AI Battle Agent, Machine Learning, Reinforcement Learning, Explainability, and Competition Deployment



This repository documents an end-to-end artificial intelligence project developed for **The Pokémon Company – PTCG AI Battle Challenge Simulation**.



The project explores data-driven and agentic decision-making for the Pokémon Trading Card Game (PTCG), progressing from card-data engineering and battle-state representation through machine-learning policy development, reinforcement-learning components, explainability, robustness testing, tournament-scale evaluation, deployment adaptation, controlled battle validation, and final competition certification.

## Project Highlights

- Built an end-to-end AI battle-agent pipeline spanning data engineering, modeling, simulation, evaluation, and deployment.
- Developed supervised ML policies with scikit-learn and PPO / reinforcement-learning components with PyTorch.
- Completed a controlled **240-battle evaluation campaign with 0 runtime failures**.
- Validated **127 / 127 sequential decisions** through the competition-engine interface.
- Implemented model explainability, feature attribution, ablation analysis, robustness testing, calibration, and behavioral analysis.
- Used SHA-256 validation, frozen artifacts, manifests, and isolated runtime tests to preserve certified deployment assets.


---
## Project Overview



The objective of this project was to build, evaluate, and validate an AI battle-agent pipeline capable of supporting strategic decision-making in a Pokémon Trading Card Game simulation environment.



Rather than treating the competition as a single-model prediction problem, the project was developed as a multi-stage AI engineering system.



The complete workflow includes:



- Pokémon card-data ingestion and normalization

- Card knowledge-base construction

- Deck representation and validation

- Battle-state representation

- Feature engineering

- Legal-action modeling

- Supervised machine-learning policies

- Reinforcement-learning / PPO components

- Policy integration

- Battle simulation

- Side-neutral evaluation

- Feature ablation

- Explainability and feature attribution

- Robustness and calibration analysis

- Tournament-scale benchmarking

- Deployment adaptation

- Controlled end-to-end battle validation

- Official-engine integration testing

- Reproducible competition packaging and certification



---
## System Architecture



```text

Raw Competition Data

        |

        v

Card Knowledge Base

        |

        v

Deck / Battle-State Representation

        |

        v

Feature Engineering

        |

        v

Legal Action Generation

        |

        +----------------------+

        |                      |

        v                      v

Supervised ML Policy       PPO / RL Policy

        |                      |

        +----------+-----------+

                   |

                   v

            Policy Integration

                   |

                   v

            Battle Simulation

                   |

                   v

       Evaluation / Benchmarking

                   |

                   v

        Explainability / Robustness

                   |

                   v

           Deployment Adapter

                   |

                   v

       Competition Submission Agent
```

## Development Roadmap
The repository contains a long-form experimental and engineering trail implemented primarily through Jupyter notebooks.



### Phase 1 — Data and Game Foundations



The initial development phase establishes:



Competition data ingestion

Card metadata processing

Card knowledge representation

Deck construction and validation

Battle-state structures

Legal-move handling

Simulator foundations

### Phase 2 — Agent and Policy Development



The project then develops:



Feature representations

Policy datasets

Supervised learning approaches

Battle-decision logic

Agent wrappers

Simulator integration

Policy evaluation

### Phase 3 — Competitive Policy Optimization



The competitive-development phase introduces:



Competitive-policy datasets

Policy training

Tournament evaluation

Elo-style benchmarking

Side-balanced search

Policy fine-tuning

Tournament-strength optimization

Legality-aware optimization

### Phase 4 — Model Validation and Explainability



The validation track includes:



Controlled feature ablation

Multi-model policy comparison

Probability attribution

Feature importance

Local explainability

Robustness analysis

Calibration and stress testing

Behavioral drift analysis

### Phase 5 — Competition Certification



The final pre-deployment phase focuses on:



Large-scale side-neutral benchmarking

Final benchmark evaluation

Certified model handoff

Packaging

Submission-interface validation

Asset-integrity verification

Deployment certification

### Phase 6 — Post-Certification Runtime Validation



The post-certification validation track extends the project without altering the frozen competition artifact.



This includes:



Corrected controlled battle scenarios

Deterministic baseline comparisons

240-battle campaign execution

Side-orientation analysis

Decision-pattern equivalence analysis

Matchup and strategic interpretation

Controlled-environment limitation analysis

## Controlled Battle Campaign
A controlled post-certification campaign was executed across:



240 battles

3 agent variants

4 controlled battle scenarios

2 side orientations

10 seeds



The evaluated agent variants included:



Deployment Policy

First Legal Move baseline

Maximum Damage Legal Move baseline



Campaign execution completed:



Requested battles : 240

Completed battles : 240

Runtime failures  : 0



The controlled campaign was used to evaluate runtime stability, action semantics, side orientation, damage and turn behavior, and decision-pattern equivalence.



Importantly, this campaign was not interpreted as proof of universal policy superiority, because the controlled environment represented a constrained attack-oriented subset of complete Pokémon TCG gameplay.



---
## Official-Engine Validation



The final deployment artifact was also exercised through the available competition-engine interface.



The recorded validation completed a terminal battle with:



```text

Accepted sequential decisions : 127 / 127

Rejected decisions            : 0



This provided end-to-end evidence that the packaged agent could participate through the expected competition interface.
```

## Model Integrity and Reproducibility
A major focus of the project was preserving validated assets and separating experimentation from certified deployment.



Later project stages use:



SHA-256 asset verification

Frozen model references

Artifact manifests

Explicit notebook-to-notebook handoffs

Controlled output directories

Backup validation

No-retraining contracts

Isolated validation environments

Post-execution integrity verification



The engineering workflow follows the principle:



```text
EXPERIMENTATION

      |

      v

VALIDATION

      |

      v

CERTIFICATION

      |

      v

FROZEN DEPLOYMENT
```



Once final deployment assets were certified, subsequent analytical work was designed not to mutate them.



## Explainability
The project includes multiple interpretability layers designed to investigate not only which actions were selected, but also the model behavior associated with those decisions.



Analyses include:



Global feature importance

Cross-strategy feature importance

Local probability attribution

Feature-ablation analysis

Probability-drift analysis

Action sensitivity

Probability redistribution

Model-behavior comparison

## Robustness and Evaluation
Evaluation extends beyond basic predictive performance.



The project includes:



Controlled battle evaluation

Side-neutral testing

Matchup analysis

Policy comparison

Stress testing

Calibration analysis

Probability drift analysis

Feature ablation

Tournament-scale evaluation

Behavioral equivalence analysis

Deployment-interface testing

## Repository Structure
PTCG\_AI\_Battle\_Challenge/

|

|-- artifacts/      # Persisted notebook evidence and validation artifacts

|-- data/           # Project data

|-- models/         # Model-related project structure

|-- notebooks/      # Experimental and development notebooks

|-- outputs/        # Generated evaluation outputs

|-- reports/        # Analysis and certification reports

|-- scripts/        # Exported and reusable project scripts

|-- src/            # Reusable project source code

|-- submission/     # Competition deployment source structure

|

|-- README.md

|-- requirements.txt

`-- .gitignore



Large competition reference files, local virtual environments, temporary files, selected model binaries, and certified archive backups may be intentionally excluded from GitHub.



## Technology Stack
Primary technologies used throughout the project include:



Python

Jupyter Notebook

pandas

NumPy

scikit-learn

PyTorch — neural-network and PPO / reinforcement-learning components

SciPy

joblib — model serialization and loading

Reinforcement-learning / PPO components

Statistical evaluation

Feature-attribution methods

Git

GitHub

SHA-256 artifact verification

## Engineering Principles
Reproducibility



Important outputs are persisted rather than relying solely on notebook state.



Traceability



Later stages explicitly reference upstream evidence and certified artifacts.



Model Integrity



Certified assets are frozen and hash-verified before and after critical validation operations.



Side Neutrality



Evaluation considers player orientation rather than assuming a single starting side.



Honest Evaluation



Controlled synthetic environments are clearly distinguished from complete official Pokémon TCG gameplay.



Deployment Validation



The project evaluates not only model behavior but also packaging, imports, entry points, legal-action handling, and runtime compatibility.



## Important Limitation
The controlled post-certification simulation campaign is not a complete reproduction of all official Pokémon TCG gameplay mechanics.



Results from that environment support claims concerning:



Runtime stability

Action semantics

Controlled decision behavior

Orientation consistency

Deterministic-baseline comparison



They should not be interpreted as proof of universal competitive superiority.



Official hidden competition evaluation remains the authoritative measure of competition performance.



## Project Status
Data pipeline                     COMPLETE

Feature engineering               COMPLETE

Policy development                COMPLETE

RL / PPO integration              COMPLETE

Simulation infrastructure         COMPLETE

Explainability                    COMPLETE

Robustness evaluation             COMPLETE

Tournament benchmarking           COMPLETE

Competition packaging             COMPLETE

Controlled 240-battle campaign    COMPLETE

Strategic interpretation          COMPLETE

Official-engine validation        COMPLETE

GitHub preservation               COMPLETE



## Author
Oluwaseyi (John) Bamigbade



AI / Machine Learning Project

The Pokémon Company – PTCG AI Battle Challenge Simulation



## Repository Note
This repository preserves the research, engineering, experimentation, validation, and deployment workflow developed during the project.



Some large competition assets, generated archives, model binaries, local runtime resources, and other non-source artifacts are intentionally excluded from version control to keep the repository manageable and to preserve appropriate separation between source history and certified deployment assets.



## Disclaimer
Pokémon and Pokémon Trading Card Game are trademarks of their respective owners.



This repository represents an independent competition and research project and is not an official product of The Pokémon Company.

