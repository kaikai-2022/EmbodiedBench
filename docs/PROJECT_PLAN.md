# Scientific Embodied Intelligence Benchmark - Project Plan

## Executive Summary

This document outlines the strategic direction for transforming the current VLABench-based project into a novel benchmark focused on **Scientific Embodied Intelligence**. The core pivot is from generic robot manipulation evaluation to systematic assessment of VLM capabilities in scientific laboratory scenarios.

## Problem Statement

Current embodied AI benchmarks (VLMbench, ALFRED, ManiSkill2) focus primarily on household and daily activity tasks. There exists a significant gap in evaluating VLM performance on **scientific laboratory manipulation tasks**, which require distinct capabilities:

- **Precision manipulation** (micrometer-level accuracy)
- **Protocol following** (multi-step experimental procedures)
- **Safety reasoning** (hazardous material handling)
- **Instrument interaction** (specialized lab equipment)
- **Measurement understanding** (quantitative reasoning)

## Research Positioning

### What This Project IS:
- A capability-oriented benchmark for scientific embodied intelligence
- A systematic evaluation of VLMs on laboratory protocol execution
- A framework for procedural scientific task generation

### What This Project IS NOT:
- Another general robot manipulation benchmark
- An incremental extension of existing household task benchmarks
- Primarily an engineering contribution

## Core Innovation Points (Reframed)

### Innovation 1: Scientific Protocol Following
**Previous framing:** "Evaluating VLMs in scientific scenarios"
**New framing:** "Systematic evaluation of protocol reasoning and execution capabilities in scientific contexts"

**Key differentiators:**
- Multi-step procedural tasks with temporal dependencies
- Quantitative measurement reasoning
- Instrument-specific interaction patterns
- Safety constraint satisfaction

### Innovation 2: Procedural Environment Generation
**Previous framing:** "Agent-based automatic scene generation"
**New framing:** "Experiment DSL and procedural task compiler for scientific embodied tasks"

**Key differentiators:**
- Task grammar for scientific experiments
- Compositional experiment templates
- Instrument library and interaction primitives
- Automatic evaluation metric generation

## Capability Taxonomy

The benchmark evaluates VLMs across multiple dimensions:

### 1. Instrument Recognition
- Identifying laboratory equipment from visual input
- Understanding instrument affordances
- Spatial relationship reasoning

### 2. Precision Manipulation
- Micrometer-level positioning accuracy
- Liquid handling (pipetting, pouring)
- Fine motor control (knob adjustment, valve operation)

### 3. Protocol Reasoning
- Multi-step procedure understanding
- Temporal dependency tracking
- Conditional execution logic
- Error recovery strategies

### 4. Measurement Understanding
- Quantitative reasoning (volumes, temperatures, time)
- Unit conversion and scaling
- Measurement precision requirements

### 5. Chemical Safety Reasoning
- Hazard identification
- Incompatibility detection
- Safety protocol adherence
- Emergency response

### 6. Spatial Alignment
- Precise object placement
- Alignment with measurement tools
- Workspace organization

## Task Categories

### Category 1: Basic Laboratory Operations
- Pipetting (volume transfer)
- Mixing solutions
- Heating/cooling operations
- Centrifugation
- Filtration

### Category 2: Instrument Interaction
- Microscope operation and adjustment
- Balance/scale usage
- pH meter operation
- Thermometer reading
- Burette titration

### Category 3: Multi-Step Protocols
- Solution preparation
- Serial dilution
- Titration experiments
- Chromatography setup
- Spectroscopy sample preparation

### Category 4: Safety-Critical Tasks
- Acid/base handling
- Volatile chemical management
- High-temperature operations
- Pressure vessel operations

## Technical Architecture

### Component 1: Experiment DSL
```
Experiment Definition
    ↓
Task Grammar Parser
    ↓
Procedural Generator
    ↓
MuJoCo Scene
```

### Component 2: Evaluation Pipeline
```
Natural Language Instruction
    ↓
VLM Processing
    ↓
Action Sequence Generation
    ↓
MuJoCo Execution
    ↓
Multi-Dimensional Evaluation
```

### Component 3: Capability Assessment
- Per-task capability scores
- Aggregate capability profiles
- Failure mode analysis
- Comparative model evaluation

## Implementation Roadmap

### Phase 1: Core Infrastructure Refinement
**Objective:** Establish robust evaluation framework

**Tasks:**
1. Refactor codebase to separate capability dimensions
2. Implement capability-specific evaluation metrics
3. Design experiment DSL syntax
4. Create instrument library with interaction primitives

### Phase 2: Task Suite Development
**Objective:** Build comprehensive scientific task collection

**Tasks:**
1. Design 50+ laboratory tasks across all categories
2. Implement procedural task generator
3. Create task difficulty taxonomy
4. Develop automatic evaluation metrics per task

### Phase 3: Capability Evaluation Framework
**Objective:** Enable multi-dimensional capability assessment

**Tasks:**
1. Implement per-capability scoring system
2. Design capability profile visualization
3. Create failure mode taxonomy
4. Build comparative analysis tools

### Phase 4: Baseline Experiments
**Objective:** Establish benchmark baselines

**Tasks:**
1. Evaluate 5-8 state-of-the-art VLMs
2. Conduct capability-wise analysis
3. Perform ablation studies
4. Document failure patterns

### Phase 5: Paper Preparation
**Objective:** Prepare NeurIPS submission

**Tasks:**
1. Write problem statement and motivation
2. Document benchmark design principles
3. Present evaluation results and analysis
4. Prepare supplementary materials

## Success Criteria

### For NeurIPS Acceptance:
1. **Clear problem definition:** Scientific embodied intelligence as distinct research area
2. **Novel capability taxonomy:** Multi-dimensional evaluation framework
3. **Comprehensive task suite:** 50+ diverse scientific tasks
4. **Rigorous evaluation:** Multiple SOTA VLMs with detailed analysis
5. **Procedural generation:** Demonstrated automatic task generation
6. **Reproducibility:** Open-source code and clear documentation

### Differentiation from Existing Work:
- **vs VLMbench:** Focus on scientific protocols vs general manipulation
- **vs VLABench:** Capability-oriented evaluation vs task planning
- **vs household benchmarks:** Precision, safety, and protocol reasoning

## Potential Paper Titles

1. **SciBench: Benchmarking Vision-Language Models for Scientific Embodied Intelligence**
2. **LabAgentBench: Evaluating VLM Capabilities in Scientific Laboratory Manipulation**
3. **From Kitchen to Laboratory: Benchmarking VLMs for Scientific Embodied Tasks**
4. **Scientific Protocol Execution: A Capability Benchmark for Embodied VLMs**

## Key Messaging for Paper

### Abstract Focus:
- Gap in scientific embodied intelligence evaluation
- Novel capability taxonomy for laboratory tasks
- Procedural task generation framework
- Comprehensive evaluation of SOTA VLMs

### Introduction Narrative:
"While embodied AI has made significant progress in household tasks, scientific laboratory manipulation remains largely unexplored. Scientific tasks require distinct capabilities: precision manipulation, protocol reasoning, safety awareness, and instrument interaction. We present [Benchmark Name], a comprehensive evaluation framework..."

### Main Contributions:
1. First systematic benchmark for scientific embodied intelligence
2. Multi-dimensional capability taxonomy for laboratory tasks
3. Procedural task generation framework with experiment DSL
4. Comprehensive evaluation revealing capability gaps in current VLMs

## Risk Mitigation

### Risk 1: Overlap with VLMbench
**Mitigation:** Emphasize scientific protocol reasoning, not general manipulation

### Risk 2: Incremental contribution perception
**Mitigation:** Focus on capability taxonomy and scientific domain novelty

### Risk 3: Limited task diversity
**Mitigation:** Ensure 50+ tasks across multiple capability dimensions

### Risk 4: Weak baseline results
**Mitigation:** Evaluate 5-8 diverse VLMs with thorough analysis

## Next Steps for Claude Code

1. **Immediate:** Review current codebase structure and identify refactoring needs
2. **Week 1-2:** Implement capability taxonomy and evaluation metrics
3. **Week 3-4:** Design and implement experiment DSL
4. **Week 5-6:** Develop task suite with procedural generation
5. **Week 7-8:** Conduct baseline experiments and analysis
6. **Week 9-10:** Paper writing and refinement

## Conclusion

This project has strong potential for NeurIPS acceptance if positioned correctly. The key is shifting from "another robot benchmark" to "systematic evaluation of scientific embodied intelligence capabilities." The scientific domain provides natural differentiation, and the capability-oriented evaluation framework aligns with current benchmark best practices.

**Core principle:** This is not about building a better system—it's about identifying and systematically evaluating a previously unexplored capability space.

---

*Document Version: 1.0*
*Last Updated: 2026-03-06*
*Target Conference: NeurIPS 2026 (May submission)*
