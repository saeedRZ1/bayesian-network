# Bayesian Network (Cloudy / Sprinkler / Rain / WetGrass)

A small Bayesian Network implementation in pure Python demonstrating exact inference by enumeration.
This project implements the classic "Cloudy / Sprinkler / Rain / WetGrass" example.

## Features
- Define BN structure and CPTs as Python functions
- Exact inference via enumeration (enumeration_ask)
- CLI for demo queries and custom queries with evidence

## Files
- `bayesian_network.py` — main script

## How to run
```bash
# Run built-in demo
python3 bayesian_network.py --demo

# Query P(Rain | WetGrass=True)
python3 bayesian_network.py --query Rain --evidence "WetGrass=True"

# Query P(Cloudy | WetGrass=True)
python3 bayesian_network.py --query Cloudy --evidence "WetGrass=True"
```

## Notes & Future improvements
- This implementation is educational: for larger networks you'd want factor-based inference, variable elimination, or sampling methods (Gibbs, rejection sampling).
- Add unit tests and more BN examples.

## License
MIT
