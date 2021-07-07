# Monty Python Markov Chain

## Description

This model is an implementation of Markov Chain that uses dialogs from *Monty Python Flying Circus* [[Kaggle]](https://www.kaggle.com/allank/monty-python-flying-circus) to predict or generate the skit sentences. 

The reasonability of the generated sentences is, however, arguable...

The model works in two modes.

### Predictive mode

The model predicts the subsequent words by means of greatest *p_value* based on the original dialogs. Unfortunately it has a chance of falling into infinite loops.

### *Generative* mode

The model predicts the subsequent words by means of uniform random value  compared with thresholds generated from the sorted cumulative *p_values*.

## Example

input:
```txt
"Hello"
"Hello" 
"The weather"
"Ah, yes... The weather"
"Indeed"
"Do"
"No"
"Goodbye"
"Goodbye" 
```

output:
```txt
- Hello, sir.
- Hello?
- The weather.
- Ah, yes... the weather.
- Indeed i take a passer by ann haydon jones the sordid details of the start with it is the arts'.
- Do you say, he's cured.
- No.
- Goodbye.
- Goodbye betty muriel sartre.
```

## Conclusion