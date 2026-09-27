# Official Engine Validation

## Purpose

This document records post-certification deployment validation performed on the completed Pokémon TCG AI Battle Challenge agent.

These tests were conducted after the research and competition artifact had been frozen. They were deployment and reproducibility checks only. They did not retrain the model or alter the historical competition submission.

## 1. Deck Request Validation

The agent was invoked using the expected observation structure for a deck request.

Observed result:

```text
Return type    : list
Returned cards : 60
All integers   : True
Status         : PASS


```

This confirmed that the agent returned a 60-card deck represented as integer card identifiers.

## 2. Official Observation Schema

The available competition API exposed the following observation structure:

```text
Observation(
    select,
    logs,
    current,
    search_begin_input
)
```

The agent successfully processed an observation generated directly by the available competition engine.

## 3. Real-Engine Battle Initialization

A battle was initialized using the available `cg.game` interface.

The engine successfully returned:

- a real observation dictionary;
- game state information;
- selection information;
- available options; and
- a valid agent decision.

The initial real-engine agent decision was returned as:

```text
Decision type : list
All integers  : True
```

## 4. Full Real-Engine Self-Play

The agent was exercised through a complete self-play battle.

Every decision was checked for:

- `list[int]` return type;
- required minimum selection count;
- allowed maximum selection count; and
- valid option-index range.

Observed result:

```text
FULL REAL-ENGINE SELF-PLAY : PASS
Terminal result            : 1
Engine selections          : 145
Decision type              : list[int]
Legality checks            : PASS
Engine cleanup             : PASS
```

The battle reached a terminal engine state without an illegal selection being detected by the validation harness.

## 5. Earlier Sequential Interface Validation

An earlier packaged-agent validation recorded:

```text
Accepted sequential decisions : 127 / 127
Rejected decisions            : 0
```

This supplied additional evidence that the packaged agent repeatedly produced accepted decisions through the expected interface.

## 6. Frozen Local Agent Archive

The locally preserved agent archive used for the later validation was:

```text
ptcg_final_agent.tar.gz
```

Recorded SHA-256:

```text
980795BF6E6A4D70C6DF7D25616670D3606625303EBAE4304F2ECADB6FA80CC8
```

This fingerprint identifies the locally validated archive.

It should not be interpreted as evidence of a new Kaggle submission or as proof that the historical competition submission was modified after submission.

## 7. Validation Scope

These checks provide deployment-level evidence for the preserved local agent artifact:

1. valid 60-card deck response;
2. compatibility with the available observation structure;
3. successful real-engine initialization;
4. legal action generation during the tested battle;
5. completion of a full self-play battle; and
6. successful engine cleanup.

They do not establish universal gameplay optimality or guarantee identical outcomes for every possible game trajectory.

---

**Project:** Pokémon TCG AI Battle Challenge  
**Validation type:** Post-certification deployment verification
